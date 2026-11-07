"""Static full-input audit; no training or intervention. Population moments, FP64."""
import csv
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'cache/cifar10-eos'
RAW = ROOT / 'runs/242-2026-11-07-mlp-all-layer-oscillation/artifacts/spark/payload/worker/inputs.npy'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summary(a):
    return dict(zip(['min', 'p05', 'median', 'p95', 'max'], map(float, np.quantile(a, [0, .05, .5, .95, 1]))))


def main():
    torch.set_num_threads(4)
    manifest = json.loads((CACHE / 'inputs.json').read_text())
    assert sha(CACHE / 'data.pt') == manifest['prepared_files']['data.pt']['sha256']
    assert sha(CACHE / 'cifar-10-python.tar.gz') == manifest['archive_sha256']
    d = torch.load(CACHE / 'data.pt', weights_only=True)
    raw = np.load(RAW)
    np.testing.assert_array_equal(raw, d['x_train'].flatten(1).numpy())
    assert hashlib.sha256(raw.tobytes()).hexdigest() == manifest['data_tensor_sha256']['x_train']
    x = raw.astype(np.float64)
    # Saved preprocessing vectors use HWC order; network Flatten uses CHW.
    means, scales = [d[k].numpy().reshape(32, 32, 3).transpose(2, 0, 1).reshape(-1)
                     for k in ['normalization_mean', 'normalization_std']]
    with tarfile.open(CACHE / 'cifar-10-python.tar.gz', 'r:gz') as bundle:
        b = pickle.load(io.BytesIO(bundle.extractfile('cifar-10-batches-py/data_batch_1').read()), encoding='bytes')
    original = b[b'data'][:5000].astype(np.float64) / 255
    np.testing.assert_array_equal(((original - means) / scales).astype(np.float32), raw)
    mu = x.mean(0)
    xc = x - mu
    sd = np.sqrt(np.mean(xc ** 2, axis=0))
    q = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1], axis=0)
    skew = np.mean(xc ** 3, axis=0) / sd ** 3
    excess = np.mean(xc ** 4, axis=0) / sd ** 4 - 3
    fields = ['column', 'channel', 'row', 'col', 'normalization_mean_01', 'normalization_std_01',
              'mean', 'std_population', 'min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max',
              'skewness', 'excess_kurtosis']
    out = HERE / 'input_column_distributions.csv'
    with out.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(fields)
        for j in range(x.shape[1]):
            writer.writerow([j, 'RGB'[j // 1024], j % 1024 // 32, j % 32, means[j], scales[j],
                             mu[j], sd[j], *q[:, j], skew[j], excess[j]])
    with threadpool_limits(limits=4):
        cov = xc.T @ xc / len(x)
        ev = np.linalg.eigvalsh(cov)[::-1]
        assert ev[-1] > 0
        corr = cov / sd[:, None] / sd[None, :]
        off = corr[np.triu_indices(x.shape[1], 1)]
        spectrum = ev / ev.sum()
        np.testing.assert_allclose(ev.sum(), np.sum(sd ** 2), rtol=1e-12)
        np.testing.assert_allclose(np.sum(ev ** 2), np.sum(cov ** 2), rtol=1e-12)
    # Independent tensor reductions validate every column mean/std and all CSV values.
    xt = torch.from_numpy(raw).double()
    np.testing.assert_allclose(mu, xt.mean(0).numpy(), atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(sd, xt.std(0, correction=0).numpy(), atol=1e-13, rtol=1e-12)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3072
    for k, a in [('mean', mu), ('std_population', sd), ('skewness', skew), ('excess_kurtosis', excess)]:
        np.testing.assert_array_equal([float(r[k]) for r in rows], a)
    for k, a in [('normalization_mean_01', means), ('normalization_std_01', scales),
                 *zip(fields[8:17], q)]:
        np.testing.assert_array_equal([float(r[k]) for r in rows], a)
    for j, row in enumerate(rows):
        assert [int(row[k]) for k in ['column', 'row', 'col']] == [j, j % 1024 // 32, j % 32]
        assert row['channel'] == 'RGB'[j // 1024]
    adjacent = []
    for c in range(3):
        for r in range(32):
            for col in range(31):
                j = c * 1024 + r * 32 + col
                adjacent.append(corr[j, j + 1])
    result = dict(status='verified', coverage=dict(samples=5000, columns=3072, ordering='CHW; RGB, row, col'),
        normalization=dict(fit_examples=50000, per_coordinate=True, population_std=True,
            raw_range=[0, 255], mean_01=summary(means), std_01=summary(scales)),
        columns=dict(mean=summary(mu), std_population=summary(sd), skewness=summary(skew),
            excess_kurtosis=summary(excess)),
        pooled=dict(min=float(x.min()), max=float(x.max()), fraction_abs_above_3=float(np.mean(abs(x) > 3)),
                    fraction_abs_above_4=float(np.mean(abs(x) > 4))),
        centered_covariance=dict(trace=float(ev.sum()), eigenvalues=ev.tolist(),
            top1_energy=float(spectrum[0]), top10_energy=float(spectrum[:10].sum()),
            top100_energy=float(spectrum[:100].sum()), participation_rank=float(1 / np.sum(spectrum ** 2)),
            top_to_mean_eigenvalue=float(ev[0] / ev.mean()), off_diagonal_correlation=summary(off),
            adjacent_horizontal_same_channel_correlation=summary(adjacent)),
        verification=dict(exact_original_bytes_to_cache=True, exact_run242_input=True,
            independent_torch_moments_columns=3072, covariance_trace_and_frobenius=True, csv_readback=True),
        sources={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
            [CACHE / 'data.pt', CACHE / 'inputs.json', CACHE / 'cifar-10-python.tar.gz', RAW,
             ROOT / 'runs/058-2026-10-13-cifar10-eos-calibration/00_prepare.py']},
        generator_sha256=sha(Path(__file__)), csv_sha256=sha(out))
    (HERE / 'input_distribution_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', newline='\n')
    result['centered_covariance'].pop('eigenvalues')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
