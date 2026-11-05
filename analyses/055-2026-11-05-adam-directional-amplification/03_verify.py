"""Independent raw-log, segmentation, overlap, aggregation and figure audit."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'analyses/036-2026-11-02-controlled-adam-diagnostics'
FEATURES = ['log_gradient', 'log_r', 'state', 'log_loss', 'loss_change4', 'drawup16']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def same(actual, expected):
    if expected is None:
        assert actual == '', actual
    else:
        np.testing.assert_allclose(float(actual), expected, rtol=2e-14, atol=1e-15)


def segment(loss):
    # Independent streaming detector; boundaries share a state but not interiors.
    opened = peak = None
    result = []
    for s in range(1, len(loss)):
        if opened is None:
            if loss[s] > loss[s - 1]:
                opened, peak = s - 1, s
        elif loss[s] <= loss[opened]:
            result.append((opened, peak, s))
            opened = peak = None
        elif loss[s] > loss[peak]:
            peak = s
    if opened is not None:
        result.append((opened, peak, None))
    return result


def main():
    manifest, plots = read(HERE / 'manifest.json'), read(HERE / 'plot_manifest.json')
    assert sha(ROOT / manifest['generator']) == manifest['generator_sha256']
    assert sha(ROOT / plots['generator']) == plots['generator_sha256']
    assert sha(HERE / 'PROTOCOL.md') == manifest['protocol_sha256']
    assert sha(HERE / 'manifest.json') == plots['reduction_manifest_sha256']
    for relative, digest in manifest['sources'].items():
        assert sha(ROOT / relative) == digest, relative
    for record in [manifest, plots]:
        for relative, digest in record['outputs'].items():
            assert sha(HERE / relative) == digest, relative
    states = table(HERE / 'derived/states.csv')
    assert len(states) == 9216
    lookup = {(r['arm'], int(r['state'])): r for r in states}
    assert len(lookup) == len(states)
    results = read(ROOT / 'runs/215-2026-11-02-controlled-full-batch-adam/results.json')
    events = table(SOURCE / 'spike_dataset/loss_spikes.csv')
    raw = {}
    eligible_count = positive_count = event_count = 0
    for condition in results['conditions']:
        arm = f'{condition["task"]}:{condition["lr"]:g}'
        folder = ROOT / condition['source']
        b, m = [[json.loads(s) for s in (folder / name).read_text().splitlines()]
                for name in ['boundaries.jsonl', 'metrics.jsonl']]
        loss = [v['loss'] for v in b]
        norms = [v['streams']['raw']['norm'] for v in m]
        raw[arm] = dict(loss=loss, relative_norm=np.array(norms) / norms[0],
                        Q=[v['projected_gradient_change'] for v in m],
                        rho=[v['streams']['raw']['cosine'] for v in m])
        found = [r for r in events if r['task'] == condition['task'] and float(r['learning_rate']) == condition['lr']]
        intervals = [(int(r['before_state']), int(r['peak_state']),
                      int(r['recovery_state']) if r['recovery_state'] else None) for r in found]
        assert intervals == segment(loss)
        event_count += len(found)
        for s in range(1024):
            r = lookup[arm, s]
            rho = m[s]['streams']['raw']['cosine']
            same(r['loss'], loss[s])
            same(r['gradient_norm'], norms[s])
            same(r['relative_norm'], norms[s] / norms[0])
            same(r['rho'], rho)
            same(r['Q'], m[s]['projected_gradient_change'])
            ratio = norms[s] / norms[s - 1] if s else None
            same(r['r'], ratio)
            same(r['log_r'], np.log(ratio) if s else None)
            same(r['log_gradient'], np.log(norms[s]))
            same(r['log_loss'], np.log(loss[s]))
            same(r['loss_change4'], np.log(loss[s] / loss[s - 4]) if s >= 4 else None)
            minimum = min(loss[max(0, s - 16):s + 1])
            same(r['drawup16'], np.log(loss[s] / minimum))
            future = max(loss[s + j] / loss[s] - 1 for j in range(1, 9)) if s + 8 <= 1024 else None
            same(r['future_max_relative_increase'], future)
            same(r['target'], int(future >= .1) if future is not None else None)
            eligible = s >= 64 and s + 8 <= 1024 and loss[s] < 1.05 * minimum
            assert int(r['eligible']) == eligible
            eligible_count += eligible
            positive_count += eligible and future >= .1
            if s:
                np.testing.assert_allclose(float(r['Q']), 1 - rho * ratio, rtol=2e-12, atol=2e-12)
    assert event_count == 812 and eligible_count == 7204 and positive_count == 645
    for summary in table(HERE / 'derived/arm_summary.csv'):
        chosen = [r for r in states if r['arm'] == summary['arm'] and r['eligible'] == '1']
        alarms = [r for r in chosen if float(r['Q']) > 2]
        same(summary['eligible'], len(chosen))
        same(summary['positives'], sum(int(r['target']) for r in chosen))
        same(summary['negative_cosines'], sum(float(r['rho']) < 0 for r in chosen))
        same(summary['growing_norms'], sum(float(r['r']) > 1 for r in chosen))
        same(summary['Q_above_two'], len(alarms))
        same(summary['Q_above_two_true'], sum(int(r['target']) for r in alarms))
        same(summary['first_loss'], raw[summary['arm']]['loss'][0])
        same(summary['final_loss'], raw[summary['arm']]['loss'][-1])
    historical = table(SOURCE / 'archive/peak_forecasts.csv')
    for r in historical:
        current = lookup[r['arm'], int(r['state'])]
        assert current['eligible'] == '1' and current['target'] == r['target']
        same(current['future_max_relative_increase'], float(r['future_max_relative_increase']))
    summaries = table(HERE / 'derived/matched_summary.csv')
    pairs = table(HERE / 'derived/matched_pairs.csv')
    calipers = np.array([manifest['calipers'][k] for k in FEATURES])
    matching_cases = 0
    for summary in summaries:
        arm, contrast = summary['arm'], summary['contrast']
        eligible = [r for r in states if r['arm'] == arm and r['eligible'] == '1' and float(r['r']) > 1]
        if contrast == 'negative_vs_nonnegative':
            more = [r for r in eligible if float(r['rho']) < 0]
            less = [r for r in eligible if float(r['rho']) >= 0]
        else:
            more = [r for r in eligible if float(r['rho']) < -.8]
            less = [r for r in eligible if -.8 <= float(r['rho']) < 0]
        selected = [p for p in pairs if p['arm'] == arm and p['contrast'] == contrast]
        n = len(selected)
        assert n == int(summary['pairs'])
        assert int(summary['eligible_more']) == len(more) and int(summary['eligible_less']) == len(less)
        more_ids = {int(r['state']) for r in more}
        less_ids = {int(r['state']) for r in less}
        assert len({p['more_state'] for p in selected}) == len({p['less_state'] for p in selected}) == n
        for p in selected:
            a, b = int(p['more_state']), int(p['less_state'])
            assert a in more_ids and b in less_ids
            x, z = lookup[arm, a], lookup[arm, b]
            for index, key in enumerate(FEATURES):
                diff = abs(float(x[key]) - float(z[key]))
                assert diff <= calipers[index] * (1 + 1e-12)
                same(p[f'abs_difference_{key}'], diff)
            assert p['more_target'] == x['target'] and p['less_target'] == z['target']
            same(p['more_future'], float(x['future_max_relative_increase']))
            same(p['less_future'], float(z['future_max_relative_increase']))
        if more and less:
            x = np.array([[float(r[k]) for k in FEATURES] for r in more])
            z = np.array([[float(r[k]) for k in FEATURES] for r in less])
            edges = np.all(np.abs(x[:, None] - z[None]) / calipers <= 1 + 1e-12, axis=2)
            # Different graph algorithm verifies maximum pair count independently.
            maximum = maximum_bipartite_matching(csr_matrix(edges), perm_type='column')
            assert np.sum(maximum >= 0) == n
            matching_cases += 1
        for prefix, group in [('more', more), ('less', less)]:
            same(summary[f'unmatched_{prefix}'], len(group) - n)
            same(summary[f'{prefix}_support'], n / len(group) if group else None)
            same(summary[f'raw_{prefix}_rate'], np.mean([int(r['target']) for r in group]) if group else None)
            same(summary[f'{prefix}_rate'], np.mean([int(p[f'{prefix}_target']) for p in selected]) if n else None)
            same(summary[f'{prefix}_mean_future'], np.mean([float(p[f'{prefix}_future']) for p in selected]) if n else None)
        same(summary['rate_difference'], np.mean([int(p['more_target']) - int(p['less_target']) for p in selected]) if n else None)
    for balance in table(HERE / 'derived/balance.csv'):
        arm, contrast, feature = balance['arm'], balance['contrast'], balance['feature']
        eligible = [r for r in states if r['arm'] == arm and r['eligible'] == '1' and float(r['r']) > 1]
        if contrast == 'negative_vs_nonnegative':
            more = [r for r in eligible if float(r['rho']) < 0]
            less = [r for r in eligible if float(r['rho']) >= 0]
        else:
            more = [r for r in eligible if float(r['rho']) < -.8]
            less = [r for r in eligible if -.8 <= float(r['rho']) < 0]
        x = np.array([float(r[feature]) for r in more])
        z = np.array([float(r[feature]) for r in less])
        selected = [p for p in pairs if p['arm'] == arm and p['contrast'] == contrast]
        differences = [float(lookup[arm, int(p['more_state'])][feature]) -
                       float(lookup[arm, int(p['less_state'])][feature]) for p in selected]
        before = float(np.mean(x) - np.mean(z))
        after = float(np.mean(differences)) if differences else None
        scale = float(np.sqrt((np.var(x) + np.var(z)) / 2))
        same(balance['mean_difference_before'], before)
        same(balance['mean_difference_after'], after)
        same(balance['standardized_before'], before / scale if scale else None)
        same(balance['standardized_after'], after / scale if scale and after is not None else None)
        same(balance['max_absolute_paired_difference'], max(map(abs, differences)) if differences else None)
    for group in table(HERE / 'derived/regimes.csv'):
        chosen = [r for r in states if r['arm'] == group['arm'] and r['eligible'] == '1'
                  and int(float(r['rho']) < 0) == int(group['negative'])
                  and int(float(r['r']) > 1) == int(group['growth'])]
        assert int(group['states']) == len(chosen)
        assert int(group['positives']) == sum(int(r['target']) for r in chosen)
    def numeric_hash(values):
        return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()
    for c in plots['curves']:
        if 'full_trajectories' in c['figure']:
            y = raw[c['arm']][c['key']]
            x = np.arange(len(y))
            if c['key'] in ['rho', 'Q']:
                x, y = x[1:], y[1:]
        elif 'direction_amplification' in c['figure']:
            chosen = [r for r in states if r['arm'] == c['arm'] and r['rho'] != '']
            if c['key'] == 'other':
                chosen = [r for r in chosen if r['eligible'] == '0']
            else:
                target = '1' if c['key'] == 'quiet_future_rise' else '0'
                chosen = [r for r in chosen if r['eligible'] == '1' and r['target'] == target]
            x, y = [float(r['rho']) for r in chosen], [float(r['r']) for r in chosen]
        else:
            r = next(s for s in summaries if s['arm'] == c['arm'] and s['contrast'] == c['contrast'])
            x, y = [100 * float(r[c['key']])], [int(r['tier'])]
        assert len(x) == c['points']
        assert numeric_hash(x) == c['x_sha256'] and numeric_hash(y) == c['y_sha256'], c
    for relative in plots['outputs']:
        with fitz.open(HERE / relative) as d:
            assert len(d) == 1 and abs(d[0].rect.width / 72 - 7.16) < .001
            assert not d[0].get_images()
            for font in d[0].get_fonts(full=True):
                assert d.extract_font(font[0])[3]
    receipt = dict(status='passed', verifier=Path(__file__).relative_to(ROOT).as_posix(),
        verifier_sha256=sha(__file__), reduction_manifest_sha256=sha(HERE / 'manifest.json'),
        plot_manifest_sha256=sha(HERE / 'plot_manifest.json'),
        checks=dict(raw_state_rows=9216, independently_segmented_events=812,
                    historical_forecast_labels=len(historical), maximum_matching_cases=matching_cases,
                    matched_pairs=len(pairs), caliper_constraints=len(pairs) * 6,
                    plotted_records=len(plots['curves']), plotted_points=sum(c['points'] for c in plots['curves']),
                    vector_embedded_font_pdfs=5),
        visual_review='See observations/003-verification-and-visual-review.md.')
    (HERE / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt['checks']))


if __name__ == '__main__':
    main()
