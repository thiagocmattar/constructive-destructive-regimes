"""Descriptive anatomy of loss excursions with an observed peak increase >10%."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'spike_anatomy'
THRESHOLD_PERCENT = 10.
TASKS = ['mlp', 'resnet', 'transformer']
LABELS = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
spec = importlib.util.spec_from_file_location('extract', HERE / '02_extract_loss_spikes.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return None if numerator is None or denominator is None or denominator == 0 else numerator / denominator


def available_mean(states, key, start, end):
    values = [states[t].get(key) for t in range(start, end + 1) if states[t].get(key) is not None]
    return None if not values else statistics.mean(values)


def features(event, loss, states):
    a, p = int(event['before_state']), int(event['peak_state'])
    c = None if event['right_censored'] else int(event['recovery_state'])
    end = int(event['end_state'])
    cross = next(t for t in range(a + 1, end + 1) if 100 * (loss[t] / loss[a] - 1) > THRESHOLD_PERCENT)
    row = dict(first_above_10pct_state=cross, before_diagnostic_state=a,
               before_diagnostic_paper_index=a + 1)
    for name, key in [('gradient', 'gradient_norm'), ('momentum', 'momentum_norm'),
                      ('adaptive', 'adam_direction_norm')]:
        before, peak = states[a].get(key), states[p].get(key)
        recovery = None if c is None else states[c].get(key)
        row[name + '_step_ratio_before'] = ratio(before, states[a - 1].get(key)) if a else None
        row[name + '_rise_ratio'] = ratio(peak, before)
        row[name + '_recovery_ratio'] = ratio(recovery, peak)
        if c is not None:
            available = [(t, states[t].get(key)) for t in range(a, c + 1) if states[t].get(key) is not None]
            tmax, vmax = max(available, key=lambda pair: pair[1])
            row[name + '_max_state_full_event'] = tmax
            row[name + '_max_lead_to_loss_peak'] = p - tmax  # Signed; negative means after the loss peak.
            row[name + '_max_amplification_full_event'] = ratio(vmax, before)
    for name in ['rho_raw', 'rho_momentum', 'rho_adam', 'eta_kappa_hat_raw']:
        # Arrival-state diagnostics describe responses over disjoint loss-transition phases.
        row[name + '_rise_mean'] = available_mean(states, name, a + 1, p)
        row[name + '_recovery_mean'] = None if c is None else available_mean(states, name, p + 1, c)
        if c is not None and name != 'eta_kappa_hat_raw':
            values = [states[t].get(name) for t in range(p + 1, c + 1) if states[t].get(name) is not None]
            row[name + '_positive_fraction_recovery'] = None if not values else sum(v > 0 for v in values) / len(values)
    row['eta_before_above_2'] = None if states[a]['eta_kappa_hat_raw'] is None else float(states[a]['eta_kappa_hat_raw'] > 2)
    row['raw_rho_before_negative'] = None if states[a].get('rho_raw') is None else float(states[a]['rho_raw'] < 0)
    first = next((t for t in range(a, end + 1)
        if states[t]['eta_kappa_hat_raw'] is not None and states[t]['eta_kappa_hat_raw'] > 2), None)
    row['first_eta_above_2_state_in_event'] = first
    for target, target_state in [('onset', a + 1), ('10pct', cross), ('loss_peak', p)]:
        row['first_eta_above_2_lead_to_' + target] = None if first is None else target_state - first
    if c is not None:
        row['loss_decrease_fraction_recovery'] = float(np.mean(np.diff(loss[p:c + 1]) < 0))
        row['loss_increase_fraction_rise'] = float(np.mean(np.diff(loss[a:p + 1]) > 0))
        available = [(t, states[t]['eta_kappa_hat_raw']) for t in range(a, c + 1)
                     if states[t]['eta_kappa_hat_raw'] is not None]
        tmax, vmax = max(available, key=lambda pair: pair[1])
        row['eta_max_full_event'] = vmax
        row['eta_max_state_full_event'] = tmax
        row['eta_max_lead_to_loss_peak'] = p - tmax
    return row


SUMMARY_METRICS = [
    'peak_change_percent', 'rise_steps', 'recovery_steps',
    'eta_kappa_hat_raw_before', 'rho_raw_before', 'rho_momentum_before', 'rho_adam_before',
    'gradient_step_ratio_before', 'gradient_rise_ratio', 'gradient_recovery_ratio',
    'momentum_rise_ratio', 'momentum_recovery_ratio', 'adaptive_rise_ratio', 'adaptive_recovery_ratio',
    'rho_raw_rise_mean', 'rho_raw_recovery_mean', 'rho_momentum_rise_mean', 'rho_momentum_recovery_mean',
    'rho_adam_rise_mean', 'rho_adam_recovery_mean',
    'eta_kappa_hat_raw_rise_mean', 'eta_kappa_hat_raw_recovery_mean',
    'rho_raw_positive_fraction_recovery', 'loss_decrease_fraction_recovery',
    'gradient_max_lead_to_loss_peak', 'momentum_max_lead_to_loss_peak', 'eta_max_lead_to_loss_peak',
    'gradient_max_amplification_full_event', 'eta_before_above_2', 'raw_rho_before_negative',
    'first_eta_above_2_lead_to_onset', 'first_eta_above_2_lead_to_10pct', 'first_eta_above_2_lead_to_loss_peak']


def summarize(all_events, task, lr, metric):
    group = all_events.loc[all_events.task.eq(task) & all_events.learning_rate.eq(lr)]
    complete = group.loc[~group.right_censored]
    values = complete[metric].dropna().to_numpy(dtype=float) if metric in complete else np.array([])
    assert np.isfinite(values).all()
    record = dict(task=task, learning_rate=lr, selected_events=len(group),
        recovered_events=len(complete), censored_events=int(group.right_censored.sum()), metric=metric, n=len(values))
    record.update(mean=None if not len(values) else float(np.mean(values)),
        std=None if len(values) < 2 else float(np.std(values, ddof=1)),
        min=None if not len(values) else float(min(values)), max=None if not len(values) else float(max(values)))
    if len(values):
        np.testing.assert_allclose(record['mean'], statistics.mean(values), rtol=1e-13, atol=1e-13)
    if len(values) > 1:
        np.testing.assert_allclose(record['std'], statistics.stdev(values), rtol=1e-13, atol=1e-13)
    return record


def write_table(summary, original):
    def get(task, lr, metric):
        return next(s for s in summary if s['task'] == task and s['learning_rate'] == lr and s['metric'] == metric)
    def value(task, lr, metric, maximum=False):
        s = get(task, lr, metric)
        if not s['n']:
            return '--'
        text = f'{s["mean"]:.2f} ± {s["std"]:.2f}' if s['std'] is not None else f'{s["mean"]:.2f} ± NA'
        return text + (f' [{s["max"]:.2f}]' if maximum else '')
    headers = ['Dataset / model', 'LR', 'N (censored)', 'Peak increase %, mean ± SD [max]',
        'Time [rise; recovery]', 'ηκ̂ [before; recovery]', 'ρ [before; recovery]', 'Gradient norm ratio [rise; recovery]']
    md = ['| ' + ' | '.join(headers) + ' |', '| --- | ---: | ---: | --- | --- | --- | --- | --- |']
    tex = ['% Observed peak increase >10%; descriptive statistics over recovered events only.',
        '% Cells: mean +/- sample SD; bracketed maximum only for peak increase.',
        '% N counts all selected events, with censored counts in parentheses.',
        '% Time and gradient-ratio pairs are [rise; recovery]; projected-change/rho pairs are [before; mean recovery].',
        r'\begin{tabular}{llrccccc}', r'\hline',
        r'Dataset/model & $\eta$ & $N$ (cens.) & Peak increase (\%) & Time $[\uparrow;\downarrow]$ & $\eta\widehat\kappa$ $[\mathrm{before};\downarrow]$ & $\rho$ $[\mathrm{before};\downarrow]$ & $\|g\|$ ratio $[\uparrow;\downarrow]$ \\', r'\hline']
    for task, label in zip(TASKS, LABELS):
        rates = sorted(original.loc[original.task.eq(task), 'learning_rate'].unique())
        for index, lr in enumerate(rates):
            s = get(task, lr, 'peak_change_percent')
            count = f'{s["selected_events"]} ({s["censored_events"]})'
            def pair(first, second):
                left, right = value(task, lr, first), value(task, lr, second)
                return '--' if left == right == '--' else '[' + left + '; ' + right + ']'
            cells = [label if index == 0 else '', f'{lr:g}', count, value(task, lr, 'peak_change_percent', True),
                pair('rise_steps', 'recovery_steps'), pair('eta_kappa_hat_raw_before', 'eta_kappa_hat_raw_recovery_mean'),
                pair('rho_raw_before', 'rho_raw_recovery_mean'), pair('gradient_rise_ratio', 'gradient_recovery_ratio')]
            md.append('| ' + ' | '.join(cells) + ' |')
            tex_cells = cells.copy()
            tex_cells[0] = r'\multirow{3}{*}{\shortstack{' + label.replace(' / ', r'\\') + '}}' if index == 0 else ''
            tex_cells = [cell.replace('±', r'$\pm$') for cell in tex_cells]
            tex.append(' & '.join(tex_cells) + r' \\')
        tex.append(r'\hline')
    tex.append(r'\end{tabular}')
    (OUT / 'table_gt10.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (OUT / 'table_gt10.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    event_path = HERE / 'spike_dataset/loss_spikes.csv'
    source_manifest = HERE / 'spike_dataset/manifest.json'
    provenance = extract.read(source_manifest)
    assert provenance['verification']['status'] == 'passed'
    assert sha(event_path) == provenance['outputs'][event_path.relative_to(HERE).as_posix()]
    original = pd.read_csv(event_path, float_precision='round_trip')
    selected = original.loc[original.peak_change_percent.gt(THRESHOLD_PERCENT)].copy()
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [event_path, source_manifest,
        HERE / '02_extract_loss_spikes.py', extract.RUN / 'results.json', extract.RUN / 'config.json']}
    for path in [extract.RUN / 'results.json', extract.RUN / 'config.json']:
        assert sources[path.relative_to(ROOT).as_posix()] == provenance['sources'][path.relative_to(ROOT).as_posix()]
    rows = []
    for condition in extract.read(extract.RUN / 'results.json')['conditions']:
        group = selected.loc[selected.task.eq(condition['task']) & selected.learning_rate.eq(condition['lr'])]
        if group.empty:
            continue
        loss, states, _ = extract.load_condition(condition, sources)
        for event in group.to_dict('records'):
            a, p = int(event['before_state']), int(event['peak_state'])
            assert event['loss_before'] == loss[a]
            assert 100 * (loss[p] / loss[a] - 1) > THRESHOLD_PERCENT
            extra = features(event, loss, states)
            if a:
                np.testing.assert_allclose(event['eta_kappa_hat_raw_before'],
                    1 - event['rho_raw_before'] * extra['gradient_step_ratio_before'], rtol=2e-12, atol=2e-12)
            c = None if event['right_censored'] else int(event['recovery_state'])
            if c is not None:
                assert extra['first_above_10pct_state'] <= p < c
                assert extra['gradient_rise_ratio'] > 0 and extra['gradient_recovery_ratio'] > 0
                assert 0 <= extra['loss_decrease_fraction_recovery'] <= 1
                # Direct arrays independently verify phase membership and gradient fold changes.
                rise_rho = [states[t].get('rho_raw') for t in range(a + 1, p + 1)]
                fall_rho = [states[t].get('rho_raw') for t in range(p + 1, c + 1)]
                np.testing.assert_allclose(extra['rho_raw_rise_mean'], np.nanmean(rise_rho), rtol=1e-13)
                np.testing.assert_allclose(extra['rho_raw_recovery_mean'], np.nanmean(fall_rho), rtol=1e-13)
                np.testing.assert_allclose(extra['gradient_rise_ratio'] * extra['gradient_recovery_ratio'],
                    states[c]['gradient_norm'] / states[a]['gradient_norm'], rtol=1e-13)
                full_gradient = [states[t]['gradient_norm'] for t in range(a, c + 1)]
                assert extra['gradient_max_state_full_event'] == a + int(np.argmax(full_gradient))
            rows.append(dict(event, **extra))
    events = pd.DataFrame(rows).sort_values(['task', 'learning_rate', 'before_state'])
    assert len(events) == len(selected) == 131
    assert set(events.spike_id) == set(selected.spike_id)
    assert sum(events.right_censored) == 2
    events.to_csv(OUT / 'events_gt10.csv', index=False)
    restored = pd.read_csv(OUT / 'events_gt10.csv', float_precision='round_trip')
    original_columns = [name for name in original.columns if name != 'spike_id']
    pd.testing.assert_frame_equal(selected.set_index('spike_id').sort_index()[original_columns],
        restored.set_index('spike_id').sort_index()[original_columns], check_dtype=False, check_exact=True)
    summary = [summarize(events, task, float(lr), metric) for task in TASKS
        for lr in sorted(original.loc[original.task.eq(task), 'learning_rate'].unique())
        for metric in SUMMARY_METRICS]
    with (OUT / 'summary_gt10.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    write_table(summary, original)
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        status='passed',
        sources=sources, selection='peak_change_percent > 10 (strict); existing event boundaries unchanged',
        selected_events=len(events), recovered_events=int((~events.right_censored).sum()), censored_events=2,
        summaries='Mean, sample SD (N-1), min and max over recovered events; per-metric n records missing values.',
        phase_clock='Rise arrival states a+1..p; recovery arrival states p+1..c; before diagnostics at state a.',
        checks=dict(source_hashes=True, original_event_values_preserved=True, before_identity=True, independent_means_and_stds=True,
            independent_phase_membership=True, signed_full_event_extrema=True, nine_task_lr_groups=True),
        outputs={p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir() if p.suffix in ['.csv', '.md', '.tex']})
    (OUT / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print((OUT / 'table_gt10.md').read_text(encoding='utf-8'))
    print('Verified 131 selected events, 129 recovered summaries, 2 censored; no figures generated.')


if __name__ == '__main__':
    main()
