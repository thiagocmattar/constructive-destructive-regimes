"""Extract loss-only excursions and join the recorded Adam diagnostics by state."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam'
OUT = HERE / 'spike_dataset'
TASKS = {'mlp': ('CIFAR-10', 'MLP'), 'resnet': ('CIFAR-10', 'ResNet-32'),
         'transformer': ('WikiText-2', 'Transformer')}
ANCHOR_METRICS = ['eta_kappa_hat_raw', 'rho_raw', 'rho_momentum', 'rho_adam',
    'gradient_norm', 'momentum_norm', 'adam_direction_norm', 'unbiased_momentum_norm',
    'momentum_over_gradient_norm', 'gradient_current_moment_cosine',
    'incoming_gradient_prior_moment_cosine', 'gradient_adaptive_cosine', 'R_adam']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excursions(loss):
    """First upward departure from a fixed pre-rise loss; no magnitude cutoff."""
    loss = np.asarray(loss, dtype=float)
    if loss.ndim != 1 or not len(loss) or not np.isfinite(loss).all():
        raise ValueError('Loss must be a nonempty finite one-dimensional sequence.')
    events, state = [], 0
    while state < len(loss) - 1:
        if loss[state + 1] <= loss[state]:
            state += 1
            continue
        before = state
        recovery = next((t for t in range(before + 2, len(loss)) if loss[t] <= loss[before]), None)
        end = recovery if recovery is not None else len(loss) - 1
        stop = recovery if recovery is not None else len(loss)
        peak = before + 1 + int(np.argmax(loss[before + 1:stop]))
        events.append(dict(before_state=before, onset_state=before + 1, peak_state=peak,
                           recovery_state=recovery, end_state=end,
                           right_censored=recovery is None))
        state = end
    return events


def flatten(record, prefix=''):
    result = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '_'))
        else:
            result[name] = value
    return result


def load_condition(condition, sources):
    folder = ROOT / condition['source']
    manifest_path = folder / 'manifest.json'
    assert sha(manifest_path) == condition['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['status'] == 'complete' and manifest['completed_updates'] == 1024
    for name in ['manifest.json', 'boundaries.jsonl', 'metrics.jsonl']:
        path = folder / name
        if name != 'manifest.json':
            assert path.stat().st_size == manifest['files'][name]['bytes']
            assert sha(path) == manifest['files'][name]['sha256']
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    boundaries = [json.loads(s) for s in (folder / 'boundaries.jsonl').read_text().splitlines()]
    metrics = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()]
    assert [b['state'] for b in boundaries] == list(range(1025))
    assert [m['update'] for m in metrics] == list(range(1, 1025))
    assert [m['gradient_state'] for m in metrics] == list(range(1024))
    loss = np.array([b['loss'] for b in boundaries])
    assert np.isfinite(loss).all() and (loss > 0).all()
    states = []
    for t, boundary in enumerate(boundaries):
        m = metrics[t] if t < len(metrics) else None
        previous = metrics[t - 1] if t else None
        state = dict(state=t, loss=boundary['loss'],
            loss_difference=None if t == 0 else loss[t] - loss[t - 1],
            gradient_norm=boundary['gradient_norm'],
            eta_kappa_hat_raw=boundary['projected_gradient_change'],
            R_adam=None if previous is None else previous['streams']['actual']['B_over_A'],
            actual_update_norm=None if previous is None else previous['streams']['actual']['norm'])
        if m:
            assert m['loss_before'] == boundary['loss']
            assert m['streams']['raw']['norm'] == boundary['gradient_norm']
            assert m['projected_gradient_change'] == boundary['projected_gradient_change']
            state.update(flatten(m))
            state.update(rho_raw=m['streams']['raw']['cosine'],
                rho_momentum=m['streams']['moment']['cosine'],
                rho_adam=m['streams']['adaptive']['cosine'],
                momentum_norm=m['streams']['moment']['norm'],
                adam_direction_norm=m['streams']['adaptive']['norm'],
                unbiased_momentum_norm=m['unbiased_moment_norm'],
                momentum_over_gradient_norm=m['moment_over_gradient_norm'])
        if t and t < len(metrics) and boundary['gradient_norm'] > 0:
            expected = 1 - state['rho_raw'] * state['gradient_norm'] / states[t - 1]['gradient_norm']
            np.testing.assert_allclose(state['eta_kappa_hat_raw'], expected, rtol=2e-12, atol=2e-12)
        states.append(state)
    return loss, states, manifest


def event_row(event, loss, states, condition, manifest, number):
    a, p, c, end = [event[k] for k in ['before_state', 'peak_state', 'recovery_state', 'end_state']]
    baseline = float(loss[a])
    after = None if c is None else float(loss[c])
    after_state = c + 1 if c is not None and c < len(loss) - 1 else None
    after_loss = None if after_state is None else float(loss[after_state])
    row = dict(spike_id=f'{condition["task"]}-lr{condition["lr"]:g}-{number:04d}',
        task=condition['task'], dataset=TASKS[condition['task']][0],
        model=TASKS[condition['task']][1], learning_rate=condition['lr'], seed=manifest['seed'],
        gpu=manifest['environment']['gpu'], objective_units=manifest['objective_units'],
        loss_units=manifest['loss_units'],
        **event, peak_is_provisional=c is None,
        loss_before=baseline, loss_onset=float(loss[a + 1]), loss_peak=float(loss[p]),
        loss_recovery=after, loss_last_observed=float(loss[end]),
        after_recovery_state=after_state, loss_after_recovery=after_loss,
        before_loss_ratio=1., peak_loss_ratio=float(loss[p] / baseline),
        recovery_loss_ratio=None if c is None else after / baseline,
        after_recovery_loss_ratio=None if after_state is None else after_loss / baseline,
        last_observed_loss_ratio=float(loss[end] / baseline),
        peak_change_percent=float(100 * (loss[p] / baseline - 1)),
        recovery_change_percent=None if c is None else 100 * (after / baseline - 1),
        after_recovery_change_percent=None if after_state is None else 100 * (after_loss / baseline - 1),
        peak_to_recovery_change_percent=None if c is None else 100 * (after / loss[p] - 1),
        peak_loss_increase=float(loss[p] - baseline), rise_steps=p - a,
        recovery_steps=None if c is None else c - p, total_steps=None if c is None else c - a,
        observed_post_peak_steps=end - p, observed_total_steps=end - a,
        excess_loss_sum=float(np.maximum(loss[a + 1:end + 1] - baseline, 0).sum()),
        source=condition['source'], initial_tensor_sha256=manifest['initial_tensor_sha256'])
    for anchor, t in [('before', a), ('peak', p), ('recovery', c)]:
        for key in ANCHOR_METRICS:
            row[f'{key}_{anchor}'] = None if t is None else states[t].get(key)
    # Retain observed extrema during the rise, without treating them as predictors.
    for key, maximum in [('eta_kappa_hat_raw', True), ('gradient_norm', True),
                         ('momentum_norm', True), ('rho_raw', False),
                         ('rho_momentum', False), ('rho_adam', False)]:
        available = [(t, states[t].get(key)) for t in range(a, p + 1) if states[t].get(key) is not None]
        best = (max if maximum else min)(available, key=lambda pair: pair[1]) if available else (None, None)
        prefix = key + ('_rise_max' if maximum else '_rise_min')
        row[prefix], row[prefix + '_state'] = best[1], best[0]
        row[prefix + '_lead_to_peak_steps'] = None if best[0] is None else p - best[0]
    return row


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_review(rows, summary):
    lines = ['# Analysis036: fixed-level loss-spike dataset', '',
        'Current work isolates loss-only excursions from all nine completed Run215 Adam histories.',
        'The fixed pre-rise reference was selected by the author. No smoothing, tolerance,',
        'magnitude cutoff, or startup exclusion is used. Earlier exploratory work stays in archive/.', '',
        '- [Mathematical definition](loss_spike_definition.md)',
        '- [Event dataset: one row per excursion](spike_dataset/loss_spikes.csv)',
        '- [All recorded event-state statistics](spike_dataset/spike_steps.csv)',
        '- [Per-run counts](spike_dataset/summary.csv) and [source hashes](spike_dataset/manifest.json)',
        '- Reproduce: `python analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py --baseline pre-rise`',
        '- Verify: `python analyses/036-2026-11-02-controlled-adam-diagnostics/03_verify_loss_spikes.py`', '',
        '## Coverage', '',
        f'{len(rows)} candidate excursions: {sum(not r["right_censored"] for r in rows)} recovered and '
        f'{sum(r["right_censored"] for r in rows)} right-censored. '
        'All 9,225 loss observations (0..1,024 per run) are searched. Counts include tiny fluctuations.', '',
        '| Model | LR | Candidates | Recovered | Censored | Largest observed peak / before |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summary:
        lines.append(f'| {TASKS[s["task"]][1]} | {s["learning_rate"]:g} | {s["excursions"]} | '
                     f'{s["recovered"]} | {s["censored"]} | {s["max_peak_loss_ratio"]:.4f} |')
    lines += ['', '## Largest observed excursion per run', '',
        'This table is a review view, ranked by peak / pre-rise loss within each run.',
        'The CSV retains all events in chronological order. Ratios and displayed losses',
        'are rounded here only; the CSV stores the full available precision.', '',
        '| Model | LR | Before / peak / recovery state | Loss before | Peak loss | Recovery loss | Ratios before / peak / recovery | Rise | Recovery |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for s in summary:
        selected = [r for r in rows if r['task'] == s['task'] and r['learning_rate'] == s['learning_rate']]
        if not selected:
            continue
        r = max(selected, key=lambda r: r['peak_loss_ratio'])
        recovery_state = 'censored' if r['right_censored'] else str(r['recovery_state'])
        loss_recovery = 'NA' if r['right_censored'] else f'{r["loss_recovery"]:.6g}'
        ratio_recovery = 'NA' if r['right_censored'] else f'{r["recovery_loss_ratio"]:.4f}'
        recovery_steps = 'NA' if r['right_censored'] else str(r['recovery_steps'])
        lines.append(f'| {r["model"]} | {r["learning_rate"]:g} | {r["before_state"]} / {r["peak_state"]} / '
            f'{recovery_state} | {r["loss_before"]:.6g} | {r["loss_peak"]:.6g} | {loss_recovery} | '
            f'1 / {r["peak_loss_ratio"]:.4f} / {ratio_recovery} | {r["rise_steps"]} | {recovery_steps} |')
    lines += ['', '## Column guide', '',
        '**Event identifiers and provenance.** `spike_id` identifies one event within a task/LR trajectory; '
        '`task`, `dataset`, `model`, `learning_rate`, `seed`, `gpu`, `objective_units`, `loss_units`, '
        '`source`, and `initial_tensor_sha256` retain the run identity. Within each case the initial weights '
        'and objective are matched across LRs. Different cloud devices were used for some arms.', '',
        '**Boundaries.** `before_state=a`, `onset_state=a+1`, `peak_state=p`, '
        '`recovery_state=c`. `end_state` is c when recovered, otherwise the observed horizon. '
        '`right_censored` and `peak_is_provisional` flag the incomplete final event. '
        '`after_recovery_state=c+1` is the next observation after the first return, when available; '
        'it makes no assertion about sustained recovery.', '',
        '**Losses and changes.** `loss_before`, `loss_onset`, `loss_peak`, `loss_recovery`, '
        '`loss_after_recovery`, and `loss_last_observed` are raw objective losses. '
        '`*_loss_ratio` divides the corresponding loss by `loss_before`; '
        '`*_change_percent` is 100 times that ratio minus 100. '
        '`peak_to_recovery_change_percent` instead divides recovery loss by peak loss. '
        '`peak_loss_increase` is the absolute peak minus pre-rise loss.', '',
        '**Durations and excess loss.** `rise_steps=p-a`, `recovery_steps=c-p`, '
        '`total_steps=c-a` count optimizer updates. `observed_post_peak_steps` and '
        '`observed_total_steps` describe follow-up to `end_state`, including censored cases. '
        '`excess_loss_sum` sums positive excess above the fixed pre-rise loss at the '
        'observed interior states, in loss units times updates. Recovery quantities remain blank '
        'when unobserved.', '',
        '**Optimizer anchors.** Each metric below has `_before`, `_peak`, and `_recovery` columns:', '',
        '| Metric prefix | Meaning |', '| --- | --- |',
        '| `eta_kappa_hat_raw` | Recorded normalized projected change of consecutive raw gradients |',
        '| `rho_raw`, `rho_momentum`, `rho_adam` | Consecutive cosine for raw gradients, stored first moments, and adaptive directions |',
        '| `gradient_norm`, `momentum_norm`, `adam_direction_norm` | Whole-vector Euclidean norms |',
        '| `unbiased_momentum_norm` | Norm of the bias-corrected first moment |',
        '| `momentum_over_gradient_norm` | Stored first-moment norm divided by raw-gradient norm |',
        '| `gradient_current_moment_cosine` | Alignment of the raw gradient with its current first moment |',
        '| `incoming_gradient_prior_moment_cosine` | Alignment of the raw gradient with the preceding first moment |',
        '| `gradient_adaptive_cosine` | Alignment of the raw gradient with the LR-free adaptive direction |',
        '| `R_adam` | Cumulative retention of applied Adam updates through the loss state |', '',
        'At parameter state s, gradient/moment/direction statistics use paper indices s+1; '
        '`R_adam` uses updates through s. No statistic is smoothed or reset at the event boundary. '
        'Terminal raw-gradient norm/projected change exist, but prospective moments/directions and '
        'their cosines do not. Undefined quantities are blank, never replaced by zero.', '',
        '**Rise-window extrema.** `*_rise_max` reports the maximum projected change or gradient/momentum '
        'norm on [a,p]; `*_rise_min` reports the minimum raw/momentum/Adam cosine on [a,p]. '
        'Each has `_state` and `_lead_to_peak_steps` columns. Earliest ties win. These are '
        'retrospective descriptors using the whole rise, including the loss peak.', '',
        '**State table.** `spike_steps.csv` retains the raw scalar record at every state from a '
        'through c (or the censored end), plus event identity, flags, loss difference, and loss ratio. '
        'Nested source keys are flattened with underscores. `streams_raw_*`, `streams_moment_*`, '
        '`streams_adaptive_*` and `streams_actual_*` preserve recorded norms, cosines, cumulative '
        'geometry, and undefined reasons; the remaining columns preserve cross-stream alignments, '
        'norm ratios, applied-step products and numerical rounding checks. These source stream '
        'prefixes include the update at s+1; `R_adam` and `actual_update_norm` separately refer '
        'to applied updates through s. Outgoing applied-step products and rounding checks require '
        'the next update and must not be used as pre-update covariates. Shared boundary states '
        'occur once per event.', '',
        '## Scope and verification', '',
        'Only existing Run215 production histories are used: three models, three constant LRs '
        'each, one frozen seed0 initialization per model, 1,024 full-objective updates per arm. '
        'MLP uses mean half squared error on its prepared 5,000 CIFAR-10 images; ResNet uses '
        'mean cross entropy on all 50,000 training images; Transformer uses token-weighted NLL '
        'on all 2,088,600 prepared targets. Loss units differ across cases.', '',
        'No new training, Hessians, LR interventions, significance tests, or causal claims. '
        'Event selection uses only loss. Event rows from the same trajectory can be dependent; '
        'these are not independent random seeds. The extraction is exploratory.', '',
        'Verification checks original file sizes/SHA-256, complete clocks, loss/gradient equality '
        'across logs, the projected-change identity, detector edge cases/ties/censoring, '
        'an independent streaming segmentation, every anchor, and every exported state statistic. '
        'The manifest pins the generator, mathematical definition and source histories.', '']
    (HERE / 'README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, choices=['pre-rise'])
    args = parser.parse_args()
    results = read(RUN / 'results.json')
    assert results['status'] == 'complete_verified' and len(results['conditions']) == 9
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in
        [RUN / 'results.json', RUN / 'config.json', HERE / 'loss_spike_definition.md',
         HERE / 'archive/loss_spike_definition_original.md']}
    rows, step_rows, summary = [], [], []
    for condition in sorted(results['conditions'], key=lambda c: (list(TASKS).index(c['task']), c['lr'])):
        loss, states, manifest = load_condition(condition, sources)
        events = excursions(loss)
        for number, event in enumerate(events, 1):
            row = event_row(event, loss, states, condition, manifest, number)
            rows.append(row)
            for t in range(event['before_state'], event['end_state'] + 1):
                step_rows.append(dict(spike_id=row['spike_id'], task=condition['task'],
                    learning_rate=condition['lr'], loss_ratio_to_before=float(loss[t] / row['loss_before']),
                    is_before=t == event['before_state'], is_peak=t == event['peak_state'],
                    is_recovery=t == event['recovery_state'], **states[t]))
        selected = [row for row in rows if row['task'] == condition['task'] and row['learning_rate'] == condition['lr']]
        summary.append(dict(task=condition['task'], learning_rate=condition['lr'], states=len(loss),
            excursions=len(events), recovered=sum(not row['right_censored'] for row in selected),
            censored=sum(row['right_censored'] for row in selected),
            max_peak_loss_ratio=max((row['peak_loss_ratio'] for row in selected), default=None)))
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / 'loss_spikes.csv', rows)
    write_csv(OUT / 'spike_steps.csv', step_rows)
    write_csv(OUT / 'summary.csv', summary)
    write_review(rows, summary)
    record = dict(generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(Path(__file__)),
        definition='Fixed pre-rise loss; first return at or below it; no magnitude threshold; no smoothing.',
        baseline=args.baseline, tolerance=0, coverage='All nine Run215 arms, all loss states 0..1024.',
        spikes=len(rows), step_rows=len(step_rows), sources=sources, summary=summary,
        outputs={p.relative_to(HERE).as_posix(): sha(p) for p in [*OUT.glob('*.csv'), HERE / 'README.md']},
        verification=dict(status='pending_independent_verifier'))
    (OUT / 'manifest.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(spikes=len(rows), step_rows=len(step_rows), summary=summary), indent=2))


if __name__ == '__main__':
    main()
