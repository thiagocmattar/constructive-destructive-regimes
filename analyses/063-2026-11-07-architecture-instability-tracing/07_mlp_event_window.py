"""Refit the audited MLP temporal Grams on outgoing updates 450--500."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / '062-2026-11-07-mlp-gradient-amplification'
spec = importlib.util.spec_from_file_location('reduce', HERE / '01_reduce.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def main():
    source = HERE / 'mlp_results.json'
    previous = json.loads(source.read_text())
    assert previous['verification']['status'] == 'passed'
    sources = {source.relative_to(ROOT).as_posix(): r.sha(source)}
    for name, digest in previous['sources'].items():
        assert r.sha(ROOT / name) == digest, name
        sources[name] = digest
    sources['analyses/063-2026-11-07-architecture-instability-tracing/01_reduce.py'] = r.sha(HERE / '01_reduce.py')

    # Original Gram rows are outgoing displacements/raw gradients 447,...,800.
    # Keep 449 as the predecessor for rho_450, then fit only 450,...,500.
    start, end, origin = 450, 500, 447
    sl = slice(start - 1 - origin, end + 1 - origin)
    all_grams = np.load(OLD / 'artifacts/all-layer-svd/grams.npz')
    gradient_grams = np.load(OLD / 'artifacts/three-space-svd/grams.npz')
    np.testing.assert_array_equal(all_grams['output'], gradient_grams['output'])
    grams = {s['name']: (gradient_grams['gradient'] if s['name'] == 'raw_gradient'
                        else all_grams[s['name']])[sl, sl] for s in previous['stages']}
    assert all(g.shape == (52, 52) for g in grams.values())
    times = np.arange(start, end + 1)
    selected = (np.array(previous['states']) >= start) & (np.array(previous['states']) <= end)
    np.testing.assert_array_equal(np.array(previous['states'])[selected], times)
    with threadpool_limits(limits=4):
        fits = {name: r.decompose(g) for name, g in grams.items()}
    output = fits['output']
    output_rho = r.cosine(grams['output'])
    stages, curves, checks = [], {}, {}
    for old_stage in previous['stages']:
        name = old_stage['name']
        g, fit = grams[name], fits[name]
        rho = r.cosine(g)
        rms = np.sqrt(np.diag(g)[1:] / old_stage['coordinates'])
        alpha = fit['coeff'][:, 0]
        old_curve = previous['curves'][name]
        np.testing.assert_allclose(rho, np.array(old_curve['rho'])[selected], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(rms, np.array(old_curve['change_rms'])[selected], atol=1e-12, rtol=1e-12)
        # Independent singular-value reduction of the Gram; no comparison to
        # the old full-window spectrum, which answers a different question.
        independent = np.linalg.svd(g[1:, 1:], compute_uv=False)
        np.testing.assert_allclose(fit['values'], independent, atol=1e-10 * independent[0], rtol=1e-7)
        residual = float(np.linalg.norm(fit['coeff'] @ fit['coeff'].T - g[1:, 1:]) / np.linalg.norm(g[1:, 1:]))
        assert residual < 1e-10
        k = min(10, fit['rank'], output['rank'])
        overlap = float(np.linalg.norm(fit['u'][:, :k].T @ output['u'][:, :k], 'fro') ** 2 / k)
        stage = copy.deepcopy(old_stage)
        stage.update(negative=int(np.sum(rho < 0)), defined=len(times),
            top1=float(fit['fractions'][0]), top10=float(fit['fractions'][:10].sum()),
            rank95=int(np.searchsorted(np.cumsum(fit['fractions']), .95) + 1),
            rank99=int(np.searchsorted(np.cumsum(fit['fractions']), .99) + 1),
            leading_sign_flips=int(np.sum(alpha[1:] * alpha[:-1] < 0)),
            correlation_with_output=float(np.corrcoef(rho, output_rho)[0, 1]),
            temporal_top10_overlap=overlap, change_rms_min=float(rms.min()), change_rms_max=float(rms.max()))
        stages.append(stage)
        curves[name] = dict(rho=rho.tolist(), change_rms=rms.tolist(), alpha=alpha.tolist(), spectrum=fit['fractions'].tolist())
        checks[name] = dict(unchanged_local_cosines_and_rms=True, independent_gram_singular_values=True,
                           gram_reconstruction_relative_error=residual, refitted_rows=len(times))

    result = copy.deepcopy(previous)
    result.update(states=times.tolist(), stages=stages, curves=curves,
        boundaries=[q for q in previous['boundaries'] if start <= q['state'] <= end],
        sources=sources, generator_sha256=r.sha(Path(__file__)),
        window=dict(start=start, end=end, svd_rows=51, gram_predecessor=449,
                    required_capture_states=[449, 501], original_gram_rows=[447, 800],
                    selection='User-requested event-scale interval; dominant peak plus smaller oscillations'),
        verification=dict(status='passed', streams=checks, output_grams_identical=True,
                          source_hashes=True, fixed_inputs=previous['input']['exactly_constant']))
    result['contract']['svd'] = 'Uncentered, no row normalization; newly fitted on 450--500 only; largest absolute temporal coefficient positive'
    peak = max(result['boundaries'], key=lambda q: q['loss'])
    result['window']['peak_state'] = peak['state']
    result['window']['peak_loss'] = peak['loss']
    r.dump(HERE / 'mlp_450_500_results.json', result)
    with (HERE / 'mlp_450_500_stages.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stages[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(stages)
    print(json.dumps(dict(window=result['window'], stages=stages), indent=2))


if __name__ == '__main__':
    main()
