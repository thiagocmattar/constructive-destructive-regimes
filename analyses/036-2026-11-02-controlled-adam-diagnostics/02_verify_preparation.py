"""Verify Run213 mechanics artifacts independently of the online capture."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json

RUN = ROOT/'runs/213-2026-11-02-eos-adam-from-init-calibration'
HERE = Path(__file__).resolve().parent


def main():
    output = RUN/'scratch/mlp-two-update-final'
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['status'] == 'smoke_complete' and manifest['mechanics_only']
    assert manifest['completed_updates'] == 2
    for name, item in manifest['files'].items():
        file = output/name
        assert file.stat().st_size == item['bytes'] and file_hash(file) == item['sha256']
    boundaries = [json.loads(line) for line in (output/'boundaries.jsonl').read_text().splitlines()]
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert [r['state'] for r in boundaries] == [0, 1, 2]
    assert [r['update'] for r in metrics] == [1, 2]
    gradients = [torch.load(output/('audit-gradient-'+str(i)+'.pt'), weights_only=True)['raw_gradient'].double() for i in range(3)]
    maximum_error = 0.
    for i, row in enumerate(boundaries):
        maximum_error = max(maximum_error, abs(float(gradients[i].norm())-row['gradient_norm']))
        if i:
            before, after = gradients[i-1], gradients[i]
            reference = 1-float(before.dot(after))/float(before.dot(before))
            assert abs(reference-row['projected_gradient_change']) < 2e-12
    assert maximum_error < 1e-10
    final = torch.load(output/'final.pt', map_location='cpu', weights_only=False)
    for state in final['optimizer']['state'].values():
        assert int(state['step']) == 2
    raw_sum = gradients[0]+gradients[1]
    assert torch.allclose(final['diagnostics']['raw']['total'], raw_sum, atol=2e-14, rtol=0)
    assert all(state['count'] == 2 for state in final['diagnostics'].values())
    evaluations = [json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert [row['state'] for row in evaluations] == [0, 2]
    assert all(row['splits']['test']['loss_units'] == 10000 for row in evaluations)
    for row in metrics:
        assert row['actual_displacement_identity_residual_norm'] < 1e-10
    inputs = json.loads((RUN/'input_manifest.json').read_text())
    for item in inputs['files']:
        file = ROOT/item['path']
        assert file.stat().st_size == item['bytes'] and file_hash(file) == item['sha256']
    result = dict(status='prepared_not_launched', scientific_arms_completed=0,
        focused_tests=dict(passed=11, failed=0,
            scope=['mixed-curvature quadratic identity', 'four exact vector prefixes against NumPy',
                'parameter telescoping', 'zero denominators', 'moment bias correction',
                'optimizer and diagnostic serialization/resume', 'uneven full-objective accumulation',
                'evaluation preserves gradients', 'loss-based regime labels',
                'failed outcomes preserved', 'tampered artifact rejected'],
            invocation='python -c: workspace TORCHINDUCTOR_CACHE_DIR, pytest scoped file, fresh run-owned --basetemp'),
        local_runtime_workarounds=['Tool child environment omits username; Torch cache explicitly placed in ignored tmp/.',
            'Existing repository default pytest temporary folder is inaccessible; used a fresh ignored run-owned scratch folder without deleting existing work.'],
        inputs=dict(files=len(inputs['files']), total_bytes=sum(x['bytes'] for x in inputs['files']),
            manifest_sha256=file_hash(RUN/'input_manifest.json'), all_sizes_and_hashes_verified=True),
        mechanics=dict(task='mlp', seed=0, lr=.0003, updates=2, objective_images=5000,
            evaluation_test_images=10000, boundary_states=[0, 1, 2], actual_stream_updates=[1, 2],
            optimizer_clocks_verified=True, stream_counts_verified=True,
            maximum_gradient_norm_cpu_gpu_error=maximum_error, projected_change_checked_independently=True,
            actual_displacement_identity_checked=True, files_verified=len(manifest['files']),
            retained_bytes=sum(item['bytes'] for item in manifest['files'].values()),
            peak_allocated_bytes=max(json.loads(line)['peak_allocated_bytes'] for line in (output/'events.jsonl').read_text().splitlines()),
            environment=manifest['environment'], source_manifest_sha256=file_hash(output/'manifest.json'),
            limitation='Mechanics only; no regime, convergence, remote speed, ResNet/Transformer fit or cross-hardware replay evidence.'),
        execution=dict(cloud_resources_created=0, new_cloud_spend=0, launch_authorization_pending=True,
            pending=['compute choice and aggregate envelope', 'production-shaped timing/fit',
                'remote start-gradient qualification', 'external guard/monitor/retrieval packet']),
        gd_milestones=dict(conditions=9, targets=36, sha256=file_hash(HERE/'gd_loss_milestones.json'), source_hashes_verified=True),
        source_sha256={p.name: file_hash(p) for p in RUN.glob('*.py')})
    write_json(HERE/'verification.json', result)
    print(json.dumps(dict(status=result['status'], tests=11, smoke_updates=2, hash_verified_files=len(inputs['files'])+len(manifest['files']))))


if __name__ == '__main__':
    main()
