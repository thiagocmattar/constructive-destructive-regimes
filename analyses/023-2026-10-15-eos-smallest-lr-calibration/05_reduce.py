"""Verify geometry and summarize the requested 3k prefixes; never edit manuscript."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp-only', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('calibration_collect', HERE / '04_collect.py')
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    checks = tools.module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'calibration_existing_checks')
    loader = tools.module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py', 'calibration_array_loader')
    cohort = json.loads((HERE / 'cohort.json').read_text())
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    results = []
    records = cohort['runs'][:1] if args.mlp_only else cohort['runs']
    for record in records:
        if record['kind'] == 'reused_gd_prefix': checks.verify_legacy(record)
        else: assert checks.verified_local(record['run'])
        d = loader.load(record)
        folder = ROOT / 'runs' / record['run'] / 'artifacts/attempt-01-spark/production'
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['initial_tensor_sha256'] == inputs['initial'][record['task']][0]['initial_tensor_sha256']
        assert manifest['config_sha256'] == file_hash(ROOT / 'runs' / record['run'] / 'config.json')
        if record['kind'] == 'new_training':
            assert manifest['full_batch'] and manifest['completed_updates'] == 3000
            assert manifest['inputs_sha256'] == file_hash(ROOT / 'cache/eos-multitask/inputs.json')
        assert len(d['loss']) == 3000 and np.isfinite(d['loss']).all()
        np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 3001, 100))
        np.testing.assert_allclose(d['q'][1:], 1 - d['cosine'][1:] * d['norm'][1:] / d['norm'][:-1], atol=2e-10, rtol=2e-10)
        assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
        rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()][:3000]
        raw = [r['streams']['raw'] for r in rows]
        get = lambda key: np.array([r[key] for r in raw], dtype=float)
        np.testing.assert_allclose(get('A'), np.cumsum(d['norm']), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('energy'), np.cumsum(d['norm'] ** 2), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('B') ** 2, get('energy') + get('all_pair'), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('lag_one'), np.r_[0, np.cumsum(2*d['cosine'][1:]*d['norm'][1:]*d['norm'][:-1])], rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(get('B_over_A'), get('B') / get('A'), rtol=2e-10, atol=2e-10)
        negative = np.flatnonzero(d['cosine'][1:] < 0)
        sustained = None
        for end in range(100, 3000):
            if np.mean(d['cosine'][end-99:end+1] < 0) >= .8:
                sustained = end+1
                break
        results.append(dict(task=record['task'], run_id=record['run_id'], lr=record['lr'], seed=0,
            updates=3000, reused_prefix=record['kind']=='reused_gd_prefix',
            initial_loss=float(d['loss'][0]), loss_at_state_3000=float(d['final_loss']),
            first_negative_gradient_update=int(negative[0]+2) if len(negative) else None,
            first_sustained_negative_100_pair_end_update=sustained,
            late_1000_cosine_mean=float(np.mean(d['cosine'][-1000:])),
            late_1000_q_mean=float(np.mean(d['q'][-1000:])),
            gradient_norm_at_state_2999=float(d['norm'][-1]), final_A=float(get('A')[-1]), final_B=float(get('B')[-1]),
            final_B_over_A=float(d['ratio'][-1]), maximum_sampled_eta_lambda=float(np.max(d['eta_lambda'])),
            last_sampled_eta_lambda=float(d['eta_lambda'][-1]),
            source_sha256=d['source_sha256'], config_sha256=manifest['config_sha256']))
    output = 'mlp_prefix_results.json' if args.mlp_only else 'results.json'
    write_json(HERE / output, dict(status='passed', utc=stamp(), runs=results,
        script_sha256=file_hash(__file__), loader_sha256=file_hash(Path(loader.__file__)),
        cohort_sha256=file_hash(HERE / 'cohort.json'),
        scope='One seed per case; smallest-LR 3000-update calibration; no replication or universal stability claim.',
        checks=['Complete used-row coverage and undefined first adjacent pair', 'Exact recorded q identity',
                'Prefix A/energy and B squared/all-pair identities', 'Lag-one sum and B/A',
                '31 residual-checked spectral states per case', 'Frozen initial tensors and source/inventory hashes'],
        gradient_pairs=len(results)*2999, spectral_points=len(results)*31,
        temporal_smoothing='none', manuscript_edits=False))
    if not args.mlp_only:
        report = ['# Three-case smallest-LR calibration', '',
            'All used scalar identities, coverage, spectral residuals and retrieved hashes pass.',
            'One seed per case, 3,000 full-GD updates; no smoothing or manuscript changes.', '',
            '| Case | LR | Loss at state3000 | Last1k mean cosine | Last1k mean q | Final B/A | First negative update |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in results:
            first = row['first_negative_gradient_update']
            report.append(f"| {row['task']} | {row['lr']:g} | {row['loss_at_state_3000']:.7g} | "
                f"{row['late_1000_cosine_mean']:.7g} | {row['late_1000_q_mean']:.7g} | "
                f"{row['final_B_over_A']:.7g} | {first if first is not None else 'Not observed'} |")
        report += ['', 'Sources and full scalar summaries: `results.json`, the frozen cohort/configs,',
            'and each owning run\'s verified raw metrics, spectra, manifest and transfer inventory.',
            'Generated by `05_reduce.py`; 8,997 adjacent pairs and93 spectral states.', '',
            'These are calibration results. One seed does not establish an LR effect or an',
            'asymptotic regime. ResNet/Transformer spectra use the declared subsets.',
            'The q/cosine relation is algebraic; neither proves loss reduction or stability.',
            'MLP reuses the Run059 prefix and is not an independent rerun.', '']
        (HERE / 'observations/002-three-case-3k.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(dict(status='passed', results=results), indent=2))


if __name__ == '__main__':
    main()
