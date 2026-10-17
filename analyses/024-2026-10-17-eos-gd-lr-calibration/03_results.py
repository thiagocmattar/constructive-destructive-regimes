"""Retrieve terminal new runs and verify all available matched 3k trajectories."""
import argparse
import functools
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    cohort = json.loads((HERE / 'cohort.json').read_text())
    checks = module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'lr_evidence_checks')
    loader = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py', 'lr_array_loader')
    if args.collect:
        remote = module(HERE / '02_launch.py', 'lr_remote')
        path = cohort['remote_root'] + '/analyses/' + HERE.name + '/queue_state.json'
        state = json.loads(remote.remote(cohort['remote_python'], f'import pathlib; print(pathlib.Path({path!r}).read_text())'))
        retriever = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/05_retrieve.py', 'lr_retriever')
        retriever.REMOTE = cohort['remote_root']
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        retriever.subprocess = SimpleNamespace(run=functools.partial(subprocess.run, creationflags=flags),
            check_output=functools.partial(subprocess.check_output, creationflags=flags))
        for r in cohort['runs']:
            if r['kind'] == 'new_training' and state['runs'].get(r['run_id'], {}).get('status') == 'complete':
                if not checks.verified_local(r['run']): retriever.retrieve(r['run'])
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    results, receipts = [], []
    for record in cohort['runs']:
        if record['kind'] == 'reused_gd_prefix': receipt = checks.verify_legacy(record)
        else:
            if not checks.verified_local(record['run']):
                assert record['kind'] == 'new_training', 'Missing reused evidence.'
                continue
            receipt = json.loads((ROOT / 'runs' / record['run'] / 'retrieval_verification.json').read_text())
        d = loader.load(record)
        folder = ROOT / 'runs' / record['run'] / 'artifacts/attempt-01-spark/production'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['initial_tensor_sha256'] == inputs['initial'][record['task']][0]['initial_tensor_sha256']
        assert manifest['config_sha256'] == record['config_sha256'] == file_hash(ROOT / 'runs' / record['run'] / 'config.json')
        if record['kind'] != 'reused_gd_prefix':
            assert manifest['full_batch'] and manifest['completed_updates'] == 3000
            assert manifest['inputs_sha256'] == file_hash(ROOT / 'cache/eos-multitask/inputs.json')
        assert len(d['loss']) == 3000 and np.isfinite(d['loss']).all() and np.isfinite(d['final_loss'])
        assert np.isfinite(d['norm']).all() and (d['norm'] > 0).all()
        np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 3001, 100))
        assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
        np.testing.assert_allclose(d['q'][1:], 1-d['cosine'][1:]*d['norm'][1:]/d['norm'][:-1], rtol=2e-10, atol=2e-10)
        rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()][:3000]
        raw = [r['streams']['raw'] for r in rows]
        get = lambda k: np.asarray([r[k] for r in raw], dtype=float)
        np.testing.assert_allclose(get('A'), np.cumsum(d['norm']), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('energy'), np.cumsum(d['norm']**2), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('B')**2, get('energy')+get('all_pair'), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('B_over_A'), get('B')/get('A'), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('lag_one'), np.r_[0, np.cumsum(2*d['cosine'][1:]*d['norm'][1:]*d['norm'][:-1])], rtol=2e-9, atol=2e-9)
        first = np.flatnonzero(d['cosine'][1:] < 0)
        results.append(dict(run_id=record['run_id'], task=record['task'], lr=record['lr'], updates=3000, seed=0,
            reused=record['kind'] != 'new_training', initial_loss=float(d['loss'][0]), final_loss=float(d['final_loss']),
            first_negative_gradient_update=int(first[0]+2) if len(first) else None,
            late_1000_cosine_mean=float(d['cosine'][-1000:].mean()), late_1000_q_mean=float(d['q'][-1000:].mean()),
            final_B_over_A=float(d['ratio'][-1]), final_raw_gradient_norm=float(d['norm'][-1]),
            maximum_sampled_eta_lambda=float(d['eta_lambda'].max()), final_sampled_eta_lambda=float(d['eta_lambda'][-1]),
            source_sha256=d['source_sha256'], config_sha256=manifest['config_sha256']))
        receipts.append(receipt)
    complete = len(results) == 9
    write_json(HERE / 'results.json', dict(status='passed' if complete else 'passed_available_conditions', utc=stamp(),
        available=len(results), planned=9, runs=results, adjacent_pairs=2999*len(results), spectral_states=31*len(results),
        script_sha256=file_hash(__file__), cohort_sha256=file_hash(HERE / 'cohort.json'),
        temporal_smoothing='none', seed_aggregation='none', manuscript_edits=False))
    write_json(HERE / 'retrieval_verification.json', dict(status='passed' if complete else 'passed_available_conditions', runs=receipts))
    lines = ['# Available 3k GD learning-rate calibrations', '',
        f'{len(results)}/9 conditions available. Same frozen seed0 within each case; no temporal smoothing.', '',
        '| Case | LR | Final loss | First negative gradient update | Last1k mean cosine | Last1k mean q | Final B/A |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(results, key=lambda x: (['mlp', 'resnet', 'transformer'].index(x['task']), x['lr'])):
        lines.append(f'| {r["task"]} | {r["lr"]:g} | {r["final_loss"]:.7g} | {r["first_negative_gradient_update"] or "Not observed"} | '
            f'{r["late_1000_cosine_mean"]:.7g} | {r["late_1000_q_mean"]:.7g} | {r["final_B_over_A"]:.7g} |')
    lines += ['', 'Sources: frozen cohort/configs, qualified loader, and hash-verified owning-run metrics, spectra,',
        'manifests and transfer inventories. Generated by `03_results.py`; scalar values and hashes in `results.json`.',
        'All used q identities, cumulative A/energy/lag-one/B-squared/B-over-A identities, 31 residual-checked',
        'spectra per condition, frozen initial tensors and source inventories pass.', '',
        'Losses cover states0..3000; gradients states0..2999; adjacent pairs end at states1..2999.',
        'MLP spectra use the full5k objective; ResNet/Transformer spectra are declared subsets.',
        'One initialization supports a conditional LR comparison, not initialization robustness or a universal stability claim.',
        'Rates were selected using earlier calibration evidence. The q/cosine correspondence is algebraic.',
        'Legacy MLP spectra requested two eigenvalues; the new solver requests one; only the largest is compared.',
        'Reused prefixes are existing evidence, not independent reruns. No manuscript changes.', '']
    (HERE / 'observations').mkdir(exist_ok=True)
    (HERE / 'observations/001-available-calibrations.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='passed' if complete else 'passed_available_conditions', available=len(results), planned=9)))


if __name__ == '__main__': main()
