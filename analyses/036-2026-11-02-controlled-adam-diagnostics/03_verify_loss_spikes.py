"""Check loss-excursion edge cases and independently audit the exported rows."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('extract', HERE / '02_extract_loss_spikes.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def check_detector():
    cases = [
        ([4, 3, 2, 1], []),
        ([1, 1, 1], []),
        ([1, 2, 1], [(0, 1, 2)]),
        ([1, 2, 3, 2, 1], [(0, 2, 4)]),
        ([1, 2, 2, 1], [(0, 1, 3)]),  # Earliest tied peak.
        ([1, 3, 2, 4, 1], [(0, 3, 4)]),  # Internal local maxima stay together.
        ([1, 2, 1, 3, 1], [(0, 1, 2), (2, 3, 4)]),
        ([3, 2, 2, 3, 2], [(2, 3, 4)]),  # Last plateau state before departure.
        ([1, 2, 3], [(0, 2, None)]),
        ([1, 3, 2], [(0, 1, None)]),
        ([2, 1, 2], [(1, 2, None)]),
        ([2], []),
    ]
    for loss, expected in cases:
        events = extract.excursions(loss)
        assert [(e['before_state'], e['peak_state'], e['recovery_state']) for e in events] == expected
    # A strictly increasing affine change of loss preserves fixed-level boundaries.
    rng = np.random.default_rng(4)
    for _ in range(25):
        values = rng.normal(size=30)
        assert extract.excursions(values) == extract.excursions(3.2 * values + 5)
    for bad in [[], [1, np.nan], [1, np.inf], [[1, 2]]]:
        try:
            extract.excursions(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid loss accepted')
    return len(cases) + 25 + 4


def independent_intervals(loss):
    """Streaming open/close state machine, separate from the extractor's search."""
    before, peak, result = None, None, []
    for t in range(1, len(loss)):
        if before is None:
            if loss[t] > loss[t - 1]:
                before, peak = t - 1, t
        elif loss[t] <= loss[before]:
            result.append((before, peak, t))
            before, peak = None, None
        elif loss[t] > loss[peak]:
            peak = t
    if before is not None:
        result.append((before, peak, None))
    return result


def main():
    detector_checks = check_detector()
    print(f'Detector: {detector_checks} edge-case / invariance checks passed.')
    output = HERE / 'spike_dataset'
    if not (output / 'manifest.json').exists():
        return
    manifest = extract.read(output / 'manifest.json')
    assert manifest['generator_sha256'] == extract.sha(HERE / '02_extract_loss_spikes.py')
    for relative, digest in manifest['sources'].items():
        assert extract.sha(extract.ROOT / relative) == digest, relative
    for relative, digest in manifest['outputs'].items():
        assert extract.sha(HERE / relative) == digest, relative
    with (output / 'loss_spikes.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == manifest['spikes'] and len({r['spike_id'] for r in rows}) == len(rows)
    point_lookup = {}
    for condition in extract.read(extract.RUN / 'results.json')['conditions']:
        loss, states, _ = extract.load_condition(condition, {})
        selected = [r for r in rows if r['task'] == condition['task'] and float(r['learning_rate']) == condition['lr']]
        actual = [(int(r['before_state']), int(r['peak_state']), int(r['recovery_state']) if r['recovery_state'] else None) for r in selected]
        assert actual == independent_intervals(loss)
        for row in selected:
            a, p, c = int(row['before_state']), int(row['peak_state']), row['recovery_state']
            c = int(c) if c else None
            end = int(row['end_state'])
            assert a < p <= end and int(row['onset_state']) == a + 1
            assert np.all(loss[a + 1:(c if c is not None else len(loss))] > loss[a])
            assert p == a + 1 + int(np.argmax(loss[a + 1:(c if c is not None else len(loss))]))
            if c is not None:
                assert loss[c] <= loss[a] and p < c
                assert int(row['total_steps']) == int(row['rise_steps']) + int(row['recovery_steps'])
            else:
                assert row['recovery_steps'] == row['total_steps'] == row['loss_recovery'] == ''
                assert row['peak_is_provisional'] == row['right_censored'] == 'True'
            for name, t in [('before', a), ('peak', p), ('recovery', c)]:
                if t is not None:
                    assert float(row['loss_' + name]) == loss[t]
                for metric in extract.ANCHOR_METRICS:
                    expected = None if t is None else states[t].get(metric)
                    found = row[f'{metric}_{name}']
                    assert (found == '') if expected is None else (float(found) == expected)
            np.testing.assert_allclose(float(row['peak_change_percent']), 100 * (loss[p] - loss[a]) / loss[a], rtol=1e-10, atol=1e-12)
            assert float(row['before_loss_ratio']) == 1
            assert float(row['peak_loss_ratio']) == loss[p] / loss[a]
            assert float(row['loss_onset']) == loss[a + 1]
            assert float(row['loss_last_observed']) == loss[end]
            assert int(row['rise_steps']) == p - a
            assert int(row['observed_total_steps']) == end - a
            assert int(row['observed_post_peak_steps']) == end - p
            np.testing.assert_allclose(float(row['excess_loss_sum']),
                sum(max(value - loss[a], 0) for value in loss[a + 1:end + 1]), rtol=2e-15, atol=1e-15)
            next_state = c + 1 if c is not None and c < len(loss) - 1 else None
            assert row['after_recovery_state'] == ('' if next_state is None else str(next_state))
            if next_state is None:
                assert row['loss_after_recovery'] == row['after_recovery_loss_ratio'] == row['after_recovery_change_percent'] == ''
            else:
                assert float(row['loss_after_recovery']) == loss[next_state]
                assert float(row['after_recovery_loss_ratio']) == loss[next_state] / loss[a]
            if c is not None:
                assert float(row['recovery_loss_ratio']) == loss[c] / loss[a]
                np.testing.assert_allclose(float(row['peak_to_recovery_change_percent']), 100 * (loss[c] - loss[p]) / loss[p], rtol=1e-10, atol=1e-12)
            for metric, kind in [('eta_kappa_hat_raw', 'max'), ('gradient_norm', 'max'),
                                 ('momentum_norm', 'max'), ('rho_raw', 'min'),
                                 ('rho_momentum', 'min'), ('rho_adam', 'min')]:
                available = [(t, states[t].get(metric)) for t in range(a, p + 1) if states[t].get(metric) is not None]
                prefix = metric + '_rise_' + kind
                if not available:
                    assert row[prefix] == row[prefix + '_state'] == row[prefix + '_lead_to_peak_steps'] == ''
                    continue
                best_value = sorted(value for _, value in available)[-1 if kind == 'max' else 0]
                best_state = next(t for t, value in available if value == best_value)
                assert float(row[prefix]) == best_value
                assert int(row[prefix + '_state']) == best_state
                assert int(row[prefix + '_lead_to_peak_steps']) == p - best_state
            point_lookup[row['spike_id']] = (a, end, loss, states)
    with (output / 'spike_steps.csv').open(encoding='utf-8', newline='') as stream:
        count = 0
        seen = set()
        for point in csv.DictReader(stream):
            event_id, t = point['spike_id'], int(point['state'])
            assert (event_id, t) not in seen
            seen.add((event_id, t))
            a, end, loss, states = point_lookup[event_id]
            assert a <= t <= end and float(point['loss']) == loss[t]
            assert float(point['loss_ratio_to_before']) == loss[t] / loss[a]
            for key, expected in states[t].items():
                found = point[key]
                assert (found == '') if expected is None else (float(found) == expected if isinstance(expected, (float, int)) else found == str(expected))
            count += 1
        assert count == manifest['step_rows'] == sum(end - a + 1 for a, end, _, _ in point_lookup.values())
    manifest['verification'] = dict(status='passed', verifier=Path(__file__).relative_to(extract.ROOT).as_posix(),
        verifier_sha256=extract.sha(Path(__file__)), detector_checks=detector_checks,
        independent_segmentation=True, all_event_anchors_and_summaries=True,
        all_exported_state_statistics=True, all_source_hashes_and_output_hashes=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Export: {len(rows)} events and {count} state rows match raw sources, independent boundaries and clocks.')


if __name__ == '__main__':
    main()
