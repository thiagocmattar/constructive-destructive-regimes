"""Refit the larger ResNet event from newly replay-qualified exact Grams."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/292-2026-11-14-resnet-major-peak-tracing'
RAW = RUN / 'artifacts/attempt-01'
NAMES = [('conv1', 'conv1.call0.output'), ('layer1', 'layer1.4.relu.call1.output'),
         ('layer2', 'layer2.4.relu.call1.output'), ('layer3', 'layer3.4.relu.call1.output'),
         ('avgpool', 'avgpool.call0.output'), ('output', 'output.log_probabilities'),
         ('raw_gradient', 'raw_gradient')]


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    r = module('01_reduce.py', 'original_reduction')
    controls = module('10_spectral_controls.py', 'spectral_controls')
    meta = json.loads((RAW / 'stages.json').read_text())
    qualification = json.loads((RAW / 'qualification.json').read_text())
    assert qualification['status'] == 'passed' and qualification['exact_loss_states'] == list(range(264, 451))
    manifest = json.loads((RAW / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    inventory = json.loads((RAW / 'inventory.json').read_text())['files']
    for item in inventory:
        if item['download']:
            p = RAW / item['path']; assert p.stat().st_size == item['bytes'] and r.sha(p) == item['sha256']
    g = dict(np.load(RAW / 'grams.npz')); levels = dict(np.load(RAW / 'levels.npz'))
    fits = {n: r.decompose(g[n]) for n in meta}
    output = fits['output']; records = []; curves = {}; fit_grams = {}
    motion = json.loads((HERE / 'resnet_390_450_motion.json').read_text())
    motion_stages = {s['name']: s for s in motion['stages']}
    boundary = r.rows(RAW / 'boundaries.jsonl'); lookup = {b['state']: b for b in boundary}
    np.testing.assert_array_equal([lookup[s]['loss'] for s in range(390, 451)], motion['loss'])
    np.testing.assert_array_equal([lookup[s]['probe_loss'] for s in range(390, 451)], motion['probe_loss'])
    for source, name in NAMES:
        gram = g[source]; f = fits[source]; info = meta[source]
        assert gram.shape == (62, 62) and len(f['fractions']) == 61
        rho = r.cosine(gram); rms = np.sqrt(np.diag(gram)[1:] / info['coordinates'])
        if source != 'raw_gradient':
            np.testing.assert_allclose(rho, motion_stages[source]['rho'], rtol=2e-10, atol=2e-12)
            np.testing.assert_allclose(rms, motion_stages[source]['change_rms'], rtol=2e-10, atol=2e-12)
        k = min(10, f['rank'], output['rank'])
        overlap = float(np.sum((f['u'][:, :k].T @ output['u'][:, :k]) ** 2) / k)
        alpha = f['coeff'][:, 0]
        record = dict(name=name, source_name=source, shape=info['shape'], coordinates=info['coordinates'],
                      top1=float(f['fractions'][0]), top10=float(f['fractions'][:10].sum()),
                      rank95=int(np.searchsorted(np.cumsum(f['fractions']), .95)+1),
                      rank99=int(np.searchsorted(np.cumsum(f['fractions']), .99)+1),
                      numerical_rank=f['rank'], leading_sign_flips=int(np.sum(alpha[1:] * alpha[:-1] < 0)),
                      temporal_top10_overlap=overlap, overlap_k=k, negative=int(np.sum(rho < 0)),
                      correlation_with_output=float(np.corrcoef(rho, r.cosine(g['output']))[0, 1]))
        assert k == 10
        records.append(record)
        curves[name] = dict(rho=rho.tolist(), change_rms=rms.tolist(), alpha=alpha.tolist(),
                            spectrum=f['fractions'].tolist(), activation_rms=np.sqrt(levels[source] / info['coordinates']).tolist())
        fit_grams[name] = gram.tolist()
    # Independent direct check of the retained raw-gradient history, without
    # reusing the remote Gram reducer or its coordinate blocking.
    raw_gradient = np.load(RAW / 'raw/raw_gradient.npy', mmap_mode='r')
    direct = np.asarray(raw_gradient[:-1], dtype=np.float64)
    with threadpool_limits(limits=4):
        independent = direct @ direct.T
    np.testing.assert_allclose(independent, g['raw_gradient'], rtol=2e-10, atol=2e-12)
    output_gram = g['output'][1:, 1:]
    diag = np.diag(output_gram); roots = np.sqrt(diag)
    spectral = dict(observed=controls.spectrum(output_gram),
                    same_energy_orthogonal=controls.spectrum(np.diag(diag)),
                    unit_row_directional=controls.spectrum(output_gram / np.outer(roots, roots)))
    np.testing.assert_allclose(controls.spectrum(output_gram[::-1, ::-1])['fractions'], spectral['observed']['fractions'], atol=1e-12)
    sources = [HERE / '01_reduce.py', HERE / '10_spectral_controls.py', HERE / 'resnet_390_450_motion.json',
               RUN / '01_worker.py', RUN / 'config.json', RUN / 'input_manifest.json', RUN / 'retrieval.json']
    sources += [RAW / n for n in ['grams.npz', 'levels.npz', 'stages.json', 'manifest.json', 'qualification.json', 'boundaries.jsonl', 'inventory.json']]
    result = dict(status='verified', case='resnet', states=list(range(389, 450)), completed_steps=list(range(390, 451)),
                  gradient_states=list(range(389, 450)), stages=records, curves=curves, temporal_grams=fit_grams,
                  boundaries=boundary, manifest=manifest, qualification=qualification,
                  spectral_controls=spectral, event=motion['event'],
                  clock='Representation row at completed step s is A_s-A_(s-1); paired raw-gradient row is g_(s-1), used by the update into s. Fit predecessor excluded.',
                  scope='Six displayed endpoints; fixed first 1000-image original BatchNorm group; full 50000-image training gradients; native unchanged seed-0 control.',
                  limitations='Retrospective fixed-window SVD; no centering, averaging, sketches or primary row normalization. Single selected event; temporal overlap is not causal propagation or identical spatial directions.',
                  verification=dict(source_and_download_hashes=True, exact_replay_losses=187, exact_initial_and_audit296_gradients=True,
                                    motion_scalar_agreement=True, independent_full_gradient_Gram=True, PSD_trace_reconstruction=True,
                                    independent_Gram_singular_values=True, row_permutation_invariance=True),
                  sources={p.relative_to(ROOT).as_posix(): r.sha(p) for p in sources}, generator_sha256=r.sha(Path(__file__)))
    r.dump(HERE / 'resnet_390_450_results.json', result)
    previous_controls = json.loads((HERE / 'spectral_controls.json').read_text())
    current_controls = dict(previous_controls)
    current_controls['cases'] = [dict(case='resnet', output_space='output.log_probabilities',
        outgoing_start=389, outgoing_end=449, rows=61, zero_rows_excluded_from_normalization=0,
        **spectral, original_output_leading_sign_flips=next(v['leading_sign_flips'] for v in records if v['name']=='output.log_probabilities'))
        if case['case']=='resnet' else case for case in previous_controls['cases']]
    current_controls.update(date='2026-11-14', generator_sha256=r.sha(Path(__file__)),
        scope='Current displayed windows: completed451--501 /390--450 /553--601. ResNet freshly refitted; MLP/Transformer controls preserved.',
        sources={p.relative_to(ROOT).as_posix():r.sha(p) for p in [HERE/'spectral_controls.json',HERE/'resnet_390_450_results.json']})
    r.dump(HERE/'spectral_controls_current.json',current_controls)
    with (HERE / 'resnet_390_450_stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    r.dump(RUN / 'verification.json', dict(status='passed', **result['verification'],
        generator_sha256=r.sha(Path(__file__)), result_sha256=r.sha(HERE / 'resnet_390_450_results.json')))
    print(json.dumps(dict(stages=records, spectral_controls=spectral), indent=2))


if __name__ == '__main__':
    main()
