"""Independent full-coverage scalar, eigenpair, checkpoint and source audit."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def vector(values):
    return np.concatenate([v.numpy().reshape(-1).astype(np.float64) for v in values])


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-9, equal_nan=True)


@torch.no_grad()
def independent_final_evaluation(parameters, data):
    """Direct affine/tanh operations,512-image chunks,FP64 output loss reduction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = {k: v.to(device) for k, v in parameters.items()}
    output = {}
    for split in ['train', 'test']:
        x, y = data['x_' + split], data['y_' + split]
        loss, correct = 0., 0
        for start in range(0, len(x), 512):
            prediction = x[start:start+512].to(device).flatten(1)
            labels = y[start:start+512].to(device)
            for index in [1, 3, 5]:
                prediction = torch.nn.functional.linear(prediction, weights[f'{index}.weight'], weights[f'{index}.bias'])
                if index != 5:
                    prediction = prediction.tanh()
            target = torch.nn.functional.one_hot(labels, 10)
            loss += float(((prediction.double() - target).square().sum() * .5))
            correct += int((prediction.argmax(1) == labels).sum())
        output[split] = {'loss': loss / len(x), 'accuracy': correct / len(x), 'examples': len(x)}
    return output


def read_run(record, cohort, input_data):
    attempt = ROOT / 'runs' / record['run'] / 'artifacts/attempt-01-spark'
    directory = attempt / 'production'
    inventory = json.loads((attempt / 'transfer_inventory.json').read_text())
    for item in inventory['files']:
        path = attempt / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed_updates'] == 20000
    assert manifest['config_sha256'] == record['config_sha256']
    assert manifest['input_sha256'] == cohort['input_manifest_sha256']
    assert manifest['initial_tensor_sha256'] == cohort['initial_tensor_sha256']
    assert manifest['train_examples'] == 5000 and manifest['test_examples'] == 10000
    for filename, identity in manifest['source_sha256'].items():
        assert sha(ROOT / 'src/cifar_eos' / filename) == identity
        assert sha(attempt / 'source/cifar_eos' / filename) == identity
    config = json.loads((directory / 'resolved_config.json').read_text())
    assert config == json.loads((ROOT / 'runs' / record['run'] / 'config.json').read_text())
    assert config['proposed_production']['accuracy_goal'] is None
    assert config['selected_optimizer'] == record['optimizer']
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').open()]
    assert len(rows) == 20000
    assert [r['update'] for r in rows] == list(range(1, 20001))
    assert [r['gradient_state_step'] for r in rows] == list(range(20000))
    assert [r['test_state_step'] for r in rows] == list(range(1, 20001))
    assert all(r['test']['examples'] == 10000 for r in rows)
    streams = list(rows[0]['streams'])
    expected = {'raw', 'displacement'}
    if record['optimizer'] == 'adam':
        expected |= {'momentum', 'unbiased_momentum', 'adaptive'}
    assert set(streams) == expected
    arrays = {'update': np.arange(1, 20001), 'train_state': np.arange(20000),
              'test_state': np.arange(1, 20001)}
    scalar = ['train_loss', 'train_accuracy', 'parameter_norm_before',
              'displacement_from_initial_norm', 'gradient_dot_update', 'raw_update_cosine',
              'raw_parameter_cosine', 'eta_kappa_proxy', 'kappa_proxy', 'momentum_raw_cosine',
              'unbiased_momentum_raw_cosine', 'momentum_to_raw_norm_ratio',
              'adaptive_to_raw_norm_ratio', 'adaptive_to_momentum_norm_ratio']
    for key in scalar:
        arrays[key] = np.array([r[key] if r[key] is not None else np.nan for r in rows])
    for key in ['loss', 'accuracy']:
        arrays['test_' + key] = np.array([r['test'][key] for r in rows])
    for name in streams:
        for key, value in rows[0]['streams'][name].items():
            if key.endswith('_reason'):
                continue
            arrays[name + '_' + key] = np.array([r['streams'][name][key] if r['streams'][name][key] is not None else np.nan for r in rows])
        a = lambda key: arrays[name + '_' + key]
        assert np.isfinite(a('norm')).all() and (a('norm') > 0).all()
        assert np.isfinite(a('cosine')[1:]).all() and np.isnan(a('cosine')[0])
        assert rows[0]['streams'][name]['cosine_undefined_reason'] == 'first_update'
        assert all(r['streams'][name]['cosine_undefined_reason'] is None for r in rows[1:])
        assert all(r['streams'][name]['ratio_undefined_reason'] is None for r in rows)
        assert np.max(np.abs(a('cosine')[1:])) <= 1 + 1e-10
        assert ((a('B_over_A') >= 0) & (a('B_over_A') <= 1 + 1e-10)).all()
        close(a('A'), np.cumsum(a('norm')))
        close(a('energy'), np.cumsum(a('norm')**2))
        close(a('lag_one'), np.r_[0., np.cumsum(2 * a('cosine')[1:] * a('norm')[1:] * a('norm')[:-1])])
        close(a('B_over_A'), a('B') / a('A'))
        close(a('A_squared'), a('A')**2)
        close(a('B_squared'), a('B')**2)
        close(a('all_pair'), a('B')**2 - a('energy'))
        close(a('beyond_lag_one'), a('all_pair') - a('lag_one'))
        close(a('triangle_gap'), a('A')**2 - a('B')**2)
    q = 1 - arrays['raw_cosine'][1:] * arrays['raw_norm'][1:] / arrays['raw_norm'][:-1]
    close(arrays['eta_kappa_proxy'][1:], q)
    close(arrays['kappa_proxy'][1:], q / record['lr'])
    close(arrays['displacement_from_initial_norm'], arrays['displacement_B'])
    if record['optimizer'] == 'adam':
        b1, b2 = config['optimizers']['adam']['betas']
        close(arrays['unbiased_momentum_norm'], arrays['momentum_norm'] / (1 - b1**arrays['update']))
        close(arrays['unbiased_momentum_cosine'], arrays['momentum_cosine'])
        close(arrays['momentum_raw_cosine'], arrays['unbiased_momentum_raw_cosine'])
        close(arrays['momentum_to_raw_norm_ratio'], arrays['momentum_norm'] / arrays['raw_norm'])
        close(arrays['adaptive_to_raw_norm_ratio'], arrays['adaptive_norm'] / arrays['raw_norm'])
        close(arrays['adaptive_to_momentum_norm_ratio'], arrays['adaptive_norm'] / arrays['momentum_norm'])
    curvature = [json.loads(line) for line in (directory / 'curvature.jsonl').open()]
    assert [r['state_step'] for r in curvature] == list(range(0, 20001, 100))
    arrays['curvature_state'] = np.array([r['state_step'] for r in curvature])
    residuals, retries, hvps = [], 0, 0
    for entry in curvature:
        for key in ['hessian', 'paper_preconditioned_hessian']:
            if key not in entry:
                continue
            spectrum = entry[key]
            assert spectrum['which'] == 'LA' and spectrum['examples'] == 5000
            assert len(spectrum['eigenvalues']) == 2
            assert spectrum['eigenvalues'][0] >= spectrum['eigenvalues'][1]
            assert np.isfinite(spectrum['eigenvalues']).all()
            assert max(spectrum['relative_residuals']) <= 0.005
            assert spectrum['solver_attempts'][-1]['status'] == 'passed'
            residuals += spectrum['relative_residuals']
            retries += len(spectrum['solver_attempts']) - 1
            hvps += spectrum['hvp_calls']
        close(entry['eta_lambda_max'], record['lr'] * entry['hessian']['eigenvalues'][0])
        if record['optimizer'] == 'gd':
            close(entry['gd_reference_2_over_eta'], 2 / record['lr'])
        elif entry['state_step'] > 0:
            assert 'paper_preconditioned_hessian' in entry
            close(entry['D_preconditioned_eigenvalues'], (1 - b1**entry['state_step']) * np.array(entry['paper_preconditioned_hessian']['eigenvalues']))
            close(entry['eta_paper_preconditioned_lambda_max'], record['lr'] * entry['paper_preconditioned_hessian']['eigenvalues'][0])
    arrays['hessian_eigenvalues'] = np.array([r['hessian']['eigenvalues'] for r in curvature])
    if record['optimizer'] == 'adam':
        arrays['preconditioned_state'] = arrays['curvature_state'][1:]
        arrays['preconditioned_eigenvalues'] = np.array([r['paper_preconditioned_hessian']['eigenvalues'] for r in curvature[1:]])
    checkpoint = torch.load(directory / 'final.pt', map_location='cpu', weights_only=False)
    initial = torch.load(ROOT / 'cache/cifar10-eos/initial.pt', weights_only=True)
    theta, theta0 = vector(checkpoint['model'].values()), vector(initial.values())
    assert checkpoint['step'] == 20000 and checkpoint['config_sha256'] == record['config_sha256']
    assert checkpoint['input_sha256'] == cohort['input_manifest_sha256']
    for name, state in checkpoint['streams'].items():
        assert state['count'] == 20000
        close(np.linalg.norm(state['previous'].numpy()), arrays[name + '_norm'][-1])
        close(np.linalg.norm(state['total'].numpy()), arrays[name + '_B'][-1])
        close(state['A'], arrays[name + '_A'][-1])
    close(checkpoint['streams']['displacement']['total'].numpy(), theta - theta0)
    state_details = {'exact_displacement_telescoping': 'passed'}
    if record['optimizer'] == 'adam':
        group = checkpoint['optimizer']['param_groups'][0]
        moments = [checkpoint['optimizer']['state'][p] for p in group['params']]
        assert all(int(s['step']) == 20000 for s in moments)
        m, v = vector([s['exp_avg'] for s in moments]), vector([s['exp_avg_sq'] for s in moments])
        mhat = m / (1 - b1**20000)
        d = mhat / (np.sqrt(v / (1 - b2**20000)) + group['eps'])
        for name, reference in [('momentum', m), ('unbiased_momentum', mhat), ('adaptive', d)]:
            close(checkpoint['streams'][name]['previous'].numpy(), reference)
        state_details['optimizer_vectors'] = 'passed'
    else:
        residual = theta - theta0 + record['lr'] * checkpoint['streams']['raw']['total'].numpy()
        state_details['gd_cumulative_fp32_update_rounding_max_abs'] = float(np.max(np.abs(residual)))
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['completed_updates'] == 20000 and summary['stop_reason'] == 'update_budget'
    assert summary['final_train']['examples'] == 5000 and summary['final_test']['examples'] == 10000
    final_digest = hashlib.sha256()
    for tensor in checkpoint['model'].values():
        final_digest.update(tensor.contiguous().numpy().tobytes())
    assert final_digest.hexdigest() == summary['final_tensor_sha256']
    close(summary['final_test']['loss'], arrays['test_loss'][-1])
    close(summary['final_test']['accuracy'], arrays['test_accuracy'][-1])
    direct = independent_final_evaluation(checkpoint['model'], input_data)
    for split in ['train', 'test']:
        np.testing.assert_allclose(direct[split]['loss'], summary['final_' + split]['loss'], rtol=5e-5, atol=2e-7)
        close(direct[split]['accuracy'], summary['final_' + split]['accuracy'])
    state_details['independent_final_forward'] = {'status': 'passed', 'results': direct,
                                                'loss_rtol': 5e-5, 'loss_atol': 2e-7, 'chunk_size': 512}
    result = dict(record, updates=20000, final_train=summary['final_train'], final_test=summary['final_test'],
                  elapsed_seconds=summary['elapsed_seconds'], memory=summary['memory'],
                  first_train_99_percent_update=next((r['update'] for r in rows if r['train_accuracy'] >= .99), None),
                  minimum_observed_train_loss=min(float(arrays['train_loss'].min()), summary['final_train']['loss']),
                  streams={}, final_hessian=arrays['hessian_eigenvalues'][-1].tolist(),
                  final_eta_lambda_max=float(record['lr'] * arrays['hessian_eigenvalues'][-1, 0]),
                  maximum_hessian_eigenvalue=float(arrays['hessian_eigenvalues'][:, 0].max()),
                  late_train_loss_mean=float(arrays['train_loss'][-1000:].mean()),
                  late_train_loss_std=float(arrays['train_loss'][-1000:].std()),
                  last_1000_eta_lambda_max_mean=float((record['lr'] * arrays['hessian_eigenvalues'][-10:, 0]).mean()),
                  eta_kappa_last_1000_mean=float(np.mean(arrays['eta_kappa_proxy'][-1000:])))
    for name in streams:
        cosine = arrays[name + '_cosine'][1:]
        result['streams'][name] = {key: float(arrays[name + '_' + key][-1]) for key in ['norm', 'A', 'B', 'B_over_A', 'energy', 'all_pair', 'lag_one', 'beyond_lag_one']}
        result['streams'][name].update(negative_adjacent_fraction=float(np.mean(cosine < 0)),
                                      first_negative_adjacent_update=int(np.flatnonzero(cosine < 0)[0] + 2) if (cosine < 0).any() else None,
                                      last_1000_negative_adjacent_fraction=float(np.mean(cosine[-1000:] < 0)),
                                      last_1000_cosine_mean=float(cosine[-1000:].mean()),
                                      minimum_norm=float(arrays[name + '_norm'].min()),
                                      final_minus_minimum_norm=float(arrays[name + '_norm'][-1] - arrays[name + '_norm'].min()))
    if record['optimizer'] == 'adam':
        result['final_preconditioned_hessian'] = arrays['preconditioned_eigenvalues'][-1].tolist()
        result['final_eta_preconditioned_lambda_max'] = float(record['lr'] * arrays['preconditioned_eigenvalues'][-1, 0])
        result['fraction_spectral_samples_above_frozen_adam_reference_38'] = float(np.mean(record['lr'] * arrays['preconditioned_eigenvalues'][:, 0] > 38))
    else:
        result['fraction_spectral_samples_eta_lambda_above_2'] = float(np.mean(record['lr'] * arrays['hessian_eigenvalues'][:, 0] > 2))
        crossed = np.flatnonzero(record['lr'] * arrays['hessian_eigenvalues'][:, 0] > 2)
        result['first_sampled_state_eta_lambda_above_2'] = int(arrays['curvature_state'][crossed[0]]) if len(crossed) else None
    npz = HERE / 'derived' / ('Run' + record['run_id'] + '.npz')
    npz.parent.mkdir(exist_ok=True)
    np.savez_compressed(npz, **arrays)
    audit = dict(run_id=record['run_id'], status='passed', raw_updates_checked=len(rows),
                 test_evaluations_checked=len(rows), eigenstates_checked=len(curvature),
                 eigenpairs_checked=len(residuals), maximum_relative_eigenpair_residual=max(residuals),
                 eigensolver_retries=retries, hvp_calls=hvps, checkpoint=state_details,
                 verified_transfer_files=len(inventory['files']),
                 verified_transfer_bytes=sum(i['bytes'] for i in inventory['files']),
                 source_metrics_sha256=sha(directory / 'metrics.jsonl'),
                 source_curvature_sha256=sha(directory / 'curvature.jsonl'),
                 derived_sha256=sha(npz), derived_bytes=npz.stat().st_size)
    save(HERE / 'results' / ('Run' + record['run_id'] + '.json'), result)
    save(HERE / 'verification' / ('Run' + record['run_id'] + '.json'), audit)
    return result, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--available', action='store_true', help='Audit only fully retrieved completed runs.')
    args = parser.parse_args()
    cohort = json.loads((HERE / 'cohort.json').read_text())
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    input_data = torch.load(ROOT / 'cache/cifar10-eos/data.pt', weights_only=True)
    records = cohort['runs']
    if args.available:
        retrieved = json.loads((HERE / 'retrieval.json').read_text())
        records = [r for r in records if r['run_id'] in retrieved['runs']]
    results, audits = [], []
    for record in records:
        result, audit = read_run(record, cohort, input_data)
        results.append(result)
        audits.append(audit)
        print(json.dumps({'run': record['run_id'], 'status': 'passed', 'final_train': result['final_train'],
                          'final_eta_lambda': result['final_eta_lambda_max']}), flush=True)
    status = 'passed' if len(results) == 8 else 'partial'
    if status == 'passed':
        initial_arrays = [dict(np.load(HERE / 'derived' / ('Run' + r['run_id'] + '.npz'))) for r in records]
        for other in initial_arrays[1:]:
            np.testing.assert_array_equal(other['hessian_eigenvalues'][0], initial_arrays[0]['hessian_eigenvalues'][0])
            np.testing.assert_array_equal(other['raw_norm'][0], initial_arrays[0]['raw_norm'][0])
            np.testing.assert_array_equal(other['train_loss'][0], initial_arrays[0]['train_loss'][0])
    save(HERE / 'results.json', {'status': status, 'runs': results})
    save(HERE / 'verification.json', {'status': status, 'utc': datetime.now(timezone.utc).isoformat(), 'runs': audits,
                                     'updates_checked': sum(r['raw_updates_checked'] for r in audits),
                                     'eigenpairs_checked': sum(r['eigenpairs_checked'] for r in audits),
                                     'limits': 'Independent scalar identities and retained final vectors; dense intermediate vectors are unavailable.'})


if __name__ == '__main__':
    main()
