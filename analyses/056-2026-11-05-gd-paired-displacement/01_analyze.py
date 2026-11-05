"""Reduce retained full-GD scalar histories; no training or source mutation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A024 = ROOT / 'analyses/024-2026-10-17-eos-gd-lr-calibration'
A035 = ROOT / 'analyses/035-2026-11-02-eos-backward-anchors'
A037 = ROOT / 'analyses/037-2026-11-02-sgd-batch-extension'
TASKS = ['mlp', 'resnet', 'transformer']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_save(name, rows):
    with (HERE / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_lengths(norm, rho):
    if np.any(np.abs(rho[1:]) > 1 + 1e-12):
        raise ValueError('Invalid cosine')
    # Avoid subtracting nearly equal squared norms for almost perfect reversal.
    return np.sqrt((norm[:-1] - norm[1:])**2 +
                   2 * norm[:-1] * norm[1:] * (1 + np.clip(rho[1:], -1, 1)))


def partitions(norm, rho, B):
    """Even prefixes: original pairs and shifted pairs with boundary singletons."""
    n = len(norm)
    assert n % 2 == 0
    lengths = pair_lengths(norm, rho)
    even = np.arange(2, n+1, 2)
    A = np.cumsum(norm)[even-1]
    H0 = np.cumsum(lengths[::2])
    H1 = norm[0] + np.r_[0., np.cumsum(lengths[1::2])] + norm[even-1]
    out = []
    for H in [H0, H1]:
        C = np.divide(H, A, out=np.full_like(H, np.nan), where=A != 0)
        D = np.divide(B[even-1], H, out=np.full_like(H, np.nan), where=H != 0)
        out.append((C, D, H))
    return even, lengths, out


def main():
    sources = {}

    def source(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    old_results = read(source(A024 / 'results.json'))
    old_runs = {r['run_id']: r for r in old_results['runs']}
    cohort = read(source(A024 / 'cohort.json'))
    arrays, summaries, endpoints, losses = {}, [], [], []
    for r in sorted(cohort['runs'], key=lambda r: (TASKS.index(r['task']), r['lr'])):
        rid = r['run_id']
        folder = ROOT / 'runs' / r['run'] / 'artifacts/attempt-01-spark/production'
        for name, digest in old_runs[rid]['source_sha256'].items():
            assert sha(source(folder / name)) == digest, (rid, name)
        all_rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()]
        rows = all_rows[:3000]
        assert [v['update'] for v in rows] == list(range(1, 3001))
        assert [v['gradient_state_step'] for v in rows] == list(range(3000))
        s = [v['streams']['raw'] for v in rows]
        norm = np.array([v['norm'] for v in s])
        rho = np.array([np.nan if v['cosine'] is None else v['cosine'] for v in s])
        A, B, energy = [np.array([v[k] for v in s]) for k in ['A', 'B', 'energy']]
        np.testing.assert_allclose(A, np.cumsum(norm), rtol=2e-13)
        np.testing.assert_allclose(energy, np.cumsum(norm**2), rtol=2e-13)
        np.testing.assert_allclose(B**2-energy, [v['all_pair'] for v in s], atol=2e-8, rtol=2e-10)
        summary = read(source(folder / 'summary.json'))
        field = 'train_loss' if 'train_loss' in rows[0] else 'full_objective_loss'
        loss = np.array([v[field] for v in rows] + [
            all_rows[3000][field] if len(all_rows) > 3000 else summary['final_train']['loss']])
        assert np.isfinite(loss).all()
        assert loss[-1] == old_runs[rid]['final_loss']
        R, Rperp, Q = B/A, np.sqrt(energy)/A, B**2/energy
        q = np.r_[np.nan, 1-rho[1:]*norm[1:]/norm[:-1]]
        even, lengths, partition = partitions(norm, rho, B)
        for key, values in dict(norm=norm, rho=rho, A=A, B=B, energy=energy,
                                loss=loss, R=R, Rperp=Rperp, Q=Q, q=q,
                                even=even, pair_lengths=lengths).items():
            arrays[f'{rid}_{key}'] = values
        for offset, (C, D, H) in enumerate(partition):
            assert np.all(C <= 1 + 1e-10) and np.all(D <= 1 + 1e-10)
            np.testing.assert_allclose(C*D, R[even-1], rtol=3e-15)
            for key, values in dict(C=C, D=D, H=H).items():
                arrays[f'{rid}_{key}{offset}'] = values
            endpoints.append(dict(run=rid, task=r['task'], lr=r['lr'], offset=offset,
                C=float(C[-1]), D=float(D[-1]), R=float(R[-1]),
                Rperp=float(Rperp[-1]), Q=float(Q[-1]),
                initial_loss=float(loss[0]), final_loss=float(loss[-1])))
            starts = np.arange(offset, 2999, 2)
            delta = loss[starts+2]-loss[starts]
            arrays[f'{rid}_loss_pair_starts{offset}'] = starts
            arrays[f'{rid}_loss_pair_delta{offset}'] = delta
            for start in [0, 2000]:
                mask = starts >= start
                v = delta[mask]
                scale = np.maximum(np.abs(loss[starts[mask]]), np.abs(loss[starts[mask]+2]))
                tol = 1e-6 * np.maximum(scale, 1e-12)
                losses.append(dict(run=rid, task=r['task'], lr=r['lr'], offset=offset,
                    window_start=start, window_end=3000, pairs=len(v),
                    decreases=int(np.sum(v<0)), increases=int(np.sum(v>0)), zeros=int(np.sum(v==0)),
                    decreases_above_tol=int(np.sum(v < -tol)), increases_above_tol=int(np.sum(v>tol)),
                    below_tol=int(np.sum(np.abs(v)<=tol)), sum_change=float(v.sum()),
                    median_change=float(np.median(v)), max_change=float(v.max()), min_change=float(v.min()),
                    covered_start=int(starts[mask][0]), covered_end=int(starts[mask][-1]+2)))
        actual_key = 'parameter_displacement_norm' if 'parameter_displacement_norm' in rows[-1] else 'displacement_from_initial_norm'
        actual = rows[-1][actual_key]
        # Both workers record this after the applied update; see frozen worker source.
        summaries.append(dict(run=rid, task=r['task'], lr=r['lr'],
            source=(folder/'metrics.jsonl').relative_to(ROOT).as_posix(),
            initial_loss=float(loss[0]), final_loss=float(loss[-1]),
            relative_loss_decrease=float(1-loss[-1]/loss[0]),
            late_rho=float(rho[-1000:].mean()), late_q=float(q[-1000:].mean()),
            late_norm_ratio=float(np.mean(norm[-1000:]/norm[-1001:-1])),
            all_pair=float(B[-1]**2-energy[-1]), lag_one=float(s[-1]['lag_one']),
            eta_B=float(r['lr']*B[-1]), actual_displacement=actual,
            displacement_relative_error=float(abs(actual-r['lr']*B[-1])/(r['lr']*B[-1])),
            cosine_clamps=int(np.sum(np.abs(rho[1:])>1))))

    lag_data = read(source(A035 / 'results.json'))
    lag_summaries = []
    for p in lag_data['profiles']:
        if p['job']['optimizer'] != 'gd':
            continue
        job = p['job']; rows = p['rows']
        assert [v['lag'] for v in rows] == list(range(1, 257))
        rho = np.array([v['cosine'] for v in rows])
        for v in rows:
            assert v['reference_state'] == p['endpoint']
            assert v['compared_state'] == p['endpoint']-v['lag']
            for backend in ['cpu', 'gpu']:
                d = v[backend]
                assert abs(d['dot']/(d['anchor_norm']*d['norm'])-v['cosine']) < 1e-10
        arrays[f"lag_{job['task']}_{p['endpoint']}"] = rho
        lag_summaries.append(dict(task=job['task'], endpoint=p['endpoint'],
            odd_negative_first32=int(np.sum(rho[:32:2]<0)), even_positive_first32=int(np.sum(rho[1:32:2]>0)),
            mean_odd_first32=float(rho[:32:2].mean()), mean_even_first32=float(rho[1:32:2].mean()),
            mean_odd_last64=float(rho[-64::2].mean()), mean_even_last64=float(rho[-63::2].mean()),
            source_job=job['id']))
    assert len(lag_summaries) == 9
    sgd = read(source(A037 / 'scientific_summary.json'))
    source(A037 / 'cells.csv')
    assert sgd['status'] == 'passed'
    source(A037 / 'results.json')
    source(ROOT / 'analyses/027-2026-10-30-minibatch-noise/results.json')
    np.savez_compressed(HERE / 'arrays.npz', **arrays)
    csv_save('endpoints.csv', endpoints)
    csv_save('loss_windows.csv', losses)
    csv_save('lag_summary.csv', lag_summaries)
    result = dict(status='computed', generator_sha256=sha(__file__), sources=sources,
        arrays_sha256=sha(HERE/'arrays.npz'), trajectories=summaries, endpoints=endpoints,
        loss_windows=losses, lag_summary=lag_summaries, sgd_cells=sgd['cells'],
        coverage=dict(gd_updates=27000, adjacent_pairs=26991, paired_partitions=18,
                      backward_profiles=9, backward_pairs=2304),
        limits='One GD initialization; shifted partitions retain two singletons. All losses are training losses. Lag profiles are separate recomputations. SGD sampling and visited states both change.')
    save(HERE / 'results.json', result)
    print(json.dumps({'endpoints':endpoints, 'high_rate_loss_windows':[x for x in losses if x['run'] in ['122','120','126']], 'lag_summary':lag_summaries}, indent=2))


if __name__ == '__main__':
    main()
