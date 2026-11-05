"""Run-local, descriptive reanalysis of the nine fixed Run215 Adam histories."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics'
RUN = ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam'
TASKS = ['mlp', 'resnet', 'transformer']
FEATURES = ['log_gradient', 'log_r', 'state', 'log_loss', 'loss_change4', 'drawup16']
CALIPERS = np.array([np.log(1.5), np.log(1.1), 128., np.log(1.25), .02, .02])


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def write_csv(path, rows):
    assert rows
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load_arms():
    pinned = read(SOURCE / 'spike_dataset/manifest.json')
    sources = {}
    paths = [HERE / 'PROTOCOL.md', SOURCE / 'spike_dataset/manifest.json',
             SOURCE / 'spike_dataset/loss_spikes.csv', SOURCE / 'loss_spike_definition.md',
             SOURCE / 'archive/PEAK_TEMPORAL_PROTOCOL.md', SOURCE / 'archive/peak_forecasts.csv',
             RUN / 'results.json', RUN / 'config.json', RUN / 'input_manifest.json']
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = sha(path)
        if relative in pinned['sources']:
            assert sources[relative] == pinned['sources'][relative], relative
    assert sha(SOURCE / 'spike_dataset/loss_spikes.csv') == pinned['outputs']['spike_dataset/loss_spikes.csv']
    results = read(RUN / 'results.json')
    assert results['status'] == 'complete_verified' and len(results['conditions']) == 9
    with (SOURCE / 'spike_dataset/loss_spikes.csv').open(encoding='utf-8', newline='') as stream:
        events = list(csv.DictReader(stream))
    arms = []
    conditions = sorted(results['conditions'], key=lambda c: (TASKS.index(c['task']), c['lr']))
    for index, condition in enumerate(conditions):
        folder = ROOT / condition['source']
        for name in ['manifest.json', 'boundaries.jsonl', 'metrics.jsonl']:
            path = folder / name
            relative = path.relative_to(ROOT).as_posix()
            sources[relative] = sha(path)
            assert sources[relative] == pinned['sources'][relative], relative
        assert sha(folder / 'manifest.json') == condition['manifest_sha256']
        identity = read(folder / 'manifest.json')
        assert identity['status'] == 'complete' and identity['completed_updates'] == 1024
        boundaries, metrics = [[json.loads(line) for line in (folder / name).read_text().splitlines()]
                               for name in ['boundaries.jsonl', 'metrics.jsonl']]
        assert [b['state'] for b in boundaries] == list(range(1025))
        assert [m['gradient_state'] for m in metrics] == list(range(1024))
        assert [m['update'] for m in metrics] == list(range(1, 1025))
        loss = np.array([b['loss'] for b in boundaries])
        gradient = np.array([m['streams']['raw']['norm'] for m in metrics])
        rho = np.array([np.nan if m['streams']['raw']['cosine'] is None else m['streams']['raw']['cosine'] for m in metrics])
        q = np.array([np.nan if m['projected_gradient_change'] is None else m['projected_gradient_change'] for m in metrics])
        np.testing.assert_array_equal(loss[:-1], [m['loss_before'] for m in metrics])
        np.testing.assert_array_equal(gradient, [b['gradient_norm'] for b in boundaries[:-1]])
        assert np.all(loss > 0) and np.all(gradient > 0)
        r = np.r_[np.nan, gradient[1:] / gradient[:-1]]
        np.testing.assert_allclose(q[1:], 1 - rho[1:] * r[1:], rtol=2e-12, atol=2e-12)
        selected = [e for e in events if e['task'] == condition['task'] and float(e['learning_rate']) == condition['lr']]
        for e in selected:
            a, p, end = [int(e[k]) for k in ['before_state', 'peak_state', 'end_state']]
            c = int(e['recovery_state']) if e['recovery_state'] else None
            assert np.all(loss[a + 1:(c if c is not None else 1025)] > loss[a])
            assert p == a + 1 + np.argmax(loss[a + 1:(c if c is not None else 1025)])
            assert c is None or loss[c] <= loss[a]
            assert float(e['loss_before']) == loss[a] and float(e['loss_peak']) == loss[p]
            assert end == (1024 if c is None else c)
        arms.append(dict(arm=f'{condition["task"]}:{condition["lr"]:g}', task=condition['task'],
                         lr=condition['lr'], tier=index % 3, source=condition['source'],
                         initial_tensor_sha256=identity['initial_tensor_sha256'],
                         loss=loss, gradient=gradient, rho=rho, r=r, Q=q, events=selected,
                         gpu=condition['environment']['gpu']))
    for task in TASKS:
        assert len({a['initial_tensor_sha256'] for a in arms if a['task'] == task}) == 1
    return arms, sources


def state_rows(arm):
    loss, g = arm['loss'], arm['gradient']
    rows = []
    for s in range(1024):
        minimum = float(min(loss[max(0, s - 16):s + 1]))
        future = float(max(loss[s + 1:s + 9]) / loss[s] - 1) if s <= 1016 else None
        eligible = s >= 64 and future is not None and loss[s] < 1.05 * minimum
        rows.append(dict(arm=arm['arm'], task=arm['task'], lr=arm['lr'], tier=arm['tier'], state=s,
            loss=float(loss[s]), gradient_norm=float(g[s]), relative_norm=float(g[s] / g[0]),
            rho=None if s == 0 else float(arm['rho'][s]),
            r=None if s == 0 else float(arm['r'][s]), Q=None if s == 0 else float(arm['Q'][s]),
            log_gradient=float(np.log(g[s])), log_r=None if s == 0 else float(np.log(arm['r'][s])),
            log_loss=float(np.log(loss[s])),
            loss_change4=None if s < 4 else float(np.log(loss[s] / loss[s - 4])),
            drawup16=float(np.log(loss[s] / minimum)), future_max_relative_increase=future,
            target=None if future is None else int(future >= .1), eligible=int(eligible)))
    return rows


def pair_context(rows, contrast):
    eligible = [r for r in rows if r['eligible'] and r['r'] > 1]
    if contrast == 'negative_vs_nonnegative':
        more = [r for r in eligible if r['rho'] < 0]
        less = [r for r in eligible if r['rho'] >= 0]
    else:
        more = [r for r in eligible if r['rho'] < -.8]
        less = [r for r in eligible if -.8 <= r['rho'] < 0]
    pairs, balance = [], []
    if more and less:
        x = np.array([[r[k] for k in FEATURES] for r in more])
        z = np.array([[r[k] for k in FEATURES] for r in less])
        differences = np.abs(x[:, None, :] - z[None, :, :]) / CALIPERS
        valid = np.all(differences <= 1 + 1e-12, axis=2)
        costs = np.sum(differences ** 2, axis=2)
        tie = np.arange(costs.size).reshape(costs.shape) / max(1, costs.size) * 1e-10
        left, right = linear_sum_assignment(np.where(valid, costs + tie, 1e6))
        chosen = [(i, j) for i, j in zip(left, right) if valid[i, j]]
        for i, j in chosen:
            a, b = more[i], less[j]
            pair = dict(arm=a['arm'], contrast=contrast, pair=len(pairs), more_state=a['state'],
                        less_state=b['state'], more_target=a['target'], less_target=b['target'],
                        more_future=a['future_max_relative_increase'], less_future=b['future_max_relative_increase'],
                        cost=float(costs[i, j]))
            pair.update({f'abs_difference_{k}': float(abs(a[k] - b[k])) for k in FEATURES})
            pairs.append(pair)
        for k, feature in enumerate(FEATURES):
            scale = float(np.sqrt((np.var(x[:, k]) + np.var(z[:, k])) / 2))
            before = float(np.mean(x[:, k]) - np.mean(z[:, k]))
            after = float(np.mean([x[i, k] - z[j, k] for i, j in chosen])) if chosen else None
            balance.append(dict(arm=rows[0]['arm'], contrast=contrast, feature=feature,
                mean_difference_before=before, mean_difference_after=after,
                standardized_before=before / scale if scale else None,
                standardized_after=after / scale if scale and after is not None else None,
                caliper=float(CALIPERS[k]), max_absolute_paired_difference=
                float(max(abs(x[i, k] - z[j, k]) for i, j in chosen)) if chosen else None))
    n = len(pairs)
    summary = dict(arm=rows[0]['arm'], task=rows[0]['task'], lr=rows[0]['lr'], tier=rows[0]['tier'],
        contrast=contrast, eligible_more=len(more), eligible_less=len(less), pairs=n,
        more_support=n / len(more) if more else None, less_support=n / len(less) if less else None,
        unmatched_more=len(more) - n, unmatched_less=len(less) - n,
        raw_more_rate=float(np.mean([r['target'] for r in more])) if more else None,
        raw_less_rate=float(np.mean([r['target'] for r in less])) if less else None,
        more_rate=float(np.mean([p['more_target'] for p in pairs])) if n else None,
        less_rate=float(np.mean([p['less_target'] for p in pairs])) if n else None,
        rate_difference=float(np.mean([p['more_target'] - p['less_target'] for p in pairs])) if n else None,
        more_mean_future=float(np.mean([p['more_future'] for p in pairs])) if n else None,
        less_mean_future=float(np.mean([p['less_future'] for p in pairs])) if n else None,
        sparse=int(n < 20))
    return summary, pairs, balance


def main():
    out = HERE / 'derived'
    out.mkdir(exist_ok=True)
    arms, sources = load_arms()
    all_states, summaries, regimes, matched, pairs, balances = [], [], [], [], [], []
    for arm in arms:
        rows = state_rows(arm)
        all_states.extend(rows)
        eligible = [r for r in rows if r['eligible']]
        alarms = [r for r in eligible if r['Q'] > 2]
        event10 = [e for e in arm['events'] if e['recovery_state'] and float(e['peak_change_percent']) > 10]
        summaries.append(dict(arm=arm['arm'], task=arm['task'], lr=arm['lr'], tier=arm['tier'],
            gpu=arm['gpu'], loss_states=1025, diagnostic_states=1024, adjacent_states=1023,
            complete_future_states=1017, censored_future_states=7, eligible=len(eligible),
            positives=sum(r['target'] for r in eligible), negative_cosines=sum(r['rho'] < 0 for r in eligible),
            growing_norms=sum(r['r'] > 1 for r in eligible), Q_above_two=len(alarms),
            Q_above_two_true=sum(r['target'] for r in alarms),
            recovered_gt10=len(event10), candidates=len(arm['events']),
            recovered=sum(bool(e['recovery_state']) for e in arm['events']),
            first_loss=float(arm['loss'][0]), final_loss=float(arm['loss'][-1])))
        for negative in [0, 1]:
            for growth in [0, 1]:
                group = [r for r in eligible if int(r['rho'] < 0) == negative and int(r['r'] > 1) == growth]
                regimes.append(dict(arm=arm['arm'], task=arm['task'], lr=arm['lr'], negative=negative,
                    growth=growth, states=len(group), positives=sum(r['target'] for r in group),
                    future_rise_rate=float(np.mean([r['target'] for r in group])) if group else None))
        for contrast in ['negative_vs_nonnegative', 'strong_vs_weak_negative']:
            summary, selected_pairs, balance = pair_context(rows, contrast)
            matched.append(summary)
            pairs.extend(selected_pairs)
            balances.extend(balance)
    # Same raw losses and eligible states as every historical forecast row.
    lookup = {(r['arm'], r['state']): r for r in all_states}
    forecast = SOURCE / 'archive/peak_forecasts.csv'
    with forecast.open(encoding='utf-8', newline='') as stream:
        historical = list(csv.DictReader(stream))
    assert len(historical) == 4534
    for row in historical:
        found = lookup[row['arm'], int(row['state'])]
        assert found['eligible'] and found['target'] == int(row['target'])
        np.testing.assert_allclose(found['future_max_relative_increase'],
                                   float(row['future_max_relative_increase']), rtol=0, atol=1e-15)
    tables = dict(states=all_states, arm_summary=summaries, regimes=regimes,
                  matched_summary=matched, matched_pairs=pairs, balance=balances)
    for name, table in tables.items():
        write_csv(out / f'{name}.csv', table)
    manifest = dict(analysis='055', scope='Exploratory descriptive step1 only; no fitted predictor.',
        generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
        sources=sources, environment=dict(numpy=np.__version__, scipy=scipy.__version__),
        protocol_sha256=sha(HERE / 'PROTOCOL.md'),
        outputs={p.relative_to(HERE).as_posix(): sha(p) for p in sorted(out.glob('*.csv'))},
        coverage=dict(arms=9, loss_states=9225, diagnostic_states=len(all_states), adjacent_states=9207,
                      eligible=sum(r['eligible'] for r in all_states),
                      eligible_positives=sum(r['target'] for r in all_states if r['eligible']),
                      historical_forecast_rows_audited=len(historical),
                      complete_future_states=9153, final_vector_states_with_incomplete_future=63),
        calipers=dict(zip(FEATURES, CALIPERS.tolist())),
        match_pairs={k: sum(s['pairs'] for s in matched if s['contrast'] == k)
                     for k in ['negative_vs_nonnegative', 'strong_vs_weak_negative']})
    dump(HERE / 'manifest.json', manifest)
    print(json.dumps(dict(coverage=manifest['coverage'], match_pairs=manifest['match_pairs'])))
    for s in matched:
        print(s['arm'], s['contrast'], f'n={s["pairs"]}', s['more_rate'], s['less_rate'])


if __name__ == '__main__':
    main()
