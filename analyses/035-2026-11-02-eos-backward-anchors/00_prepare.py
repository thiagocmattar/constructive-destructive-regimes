"""Freeze the nine source states,18 optimizer arms and per-worker input packets."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RUN = ROOT/'runs/212-2026-11-02-eos-backward-anchors'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    assert not (RUN/'input_manifest.json').exists(), 'Inputs already frozen'
    old = json.loads((ROOT/'runs/210-2026-11-01-eos-adam-long-lags/config.json').read_text())
    parent = ROOT/'runs/210-2026-11-01-eos-adam-long-lags/input_manifest.json'
    expected = {x['path']:x for x in json.loads(parent.read_text())['files']}
    jobs = []
    for task, rid, base in [('transformer','126',0),('resnet','120',6),('mlp','122',12)]:
        for i, start in enumerate([500,2000,3000]):
            source = next(j for j in old['jobs'] if j['parent_run_id']==rid and j['anchor']==start)
            for n, optimizer in enumerate(['gd','adam']):
                job = dict(source)
                for key in ['rates','horizon','production_horizon','worker','anchor']:
                    job.pop(key, None)
                job.update(id=f'{optimizer}-{rid}-{start}', start=start, endpoint=start+256,
                           optimizer=optimizer, lr=source['parent_gd_lr'] if optimizer=='gd' else source['rates'][0],
                           worker=12 if task=='mlp' else base+2*i+n)
                assert job['primary_eligible']
                jobs.append(job)
    cfg = dict(run_id='212', analysis='035', horizon=256, jobs=jobs,
        metric='rho_t(k)=cos(g_t,g_(t-k)); fixed endpoint t=start+256, k=1..256',
        objective=old['objective'], parameter_scope=old['parameter_scope'], precision=old['precision'],
        zero_denominator=old['zero_denominator'], validation=old['validation'],
        adam=old['optimizer'], gd=dict(name='SGD', momentum=0, weight_decay=0, foreach=False, schedule='constant'),
        task_cache=old['task_cache'], original_inputs=old['original_inputs'],
        inputs_manifest=RUN.relative_to(ROOT).as_posix()+'/input_manifest.json',
        stop_request=RUN.relative_to(ROOT).as_posix()+'/stop-request',
        qualification=dict(saved_gradient_relative=1e-5, chunk_gradient_relative=1e-5, loss_absolute=1e-5,
                           cosine_cpu_gpu_absolute=1e-10),
        retention='All scalar products and CPU checks; every raw-gradient SHA; full gradients at offsets0,128,192,254,255,256; terminal model/optimizer/RNG. Dense257-gradient temporary histories removed only after reduction/audits are saved.',
        source_qualification='Identical directly saved initial tensors; qualify each hardware/chunk against saved full gradient. No claim of bitwise replay of later historical trajectories.',
        authorization='User requests backward anchors, selects later Adam endpoints, requests separate GD/Adam figures in new analysis, only256 lags and fastest parallel execution. Prior Runpod funding and terminate-all-task-pods authorization persists.',
        limits=dict(max_total_usd=15, graceful_compute_stop_usd=12.5, hard_compute_stop_usd=14,
                    max_lease_seconds=7200, max_concurrent_gpus=13, container_gb=60, pod_volume_gb=35),
        inference='One seed per architecture; dependent forks; no uncertainty bands, stationary assumption, stability, generalization or optimizer-superiority claim.')
    save(RUN/'config.json', cfg)
    files = {}
    def add(path, workers):
        files.setdefault(path, set()).update(workers)
    workers = list(range(13))
    for job in jobs:
        for field in ['checkpoint','anchor_gradient','data']:
            add(job[field], [job['worker']])
    shared = [p for p in expected if p.startswith(('src/','cache/eos-multitask/reference/'))]
    shared += [old['original_inputs'], (RUN/'config.json').relative_to(ROOT).as_posix(),
               (RUN/'01_worker.py').relative_to(ROOT).as_posix()]
    for path in shared:
        add(path, workers)
    inventory = []
    for rel, assigned in sorted(files.items()):
        path = ROOT/rel
        digest = sha(path)
        if rel in expected:
            assert digest == expected[rel]['sha256'] and path.stat().st_size == expected[rel]['bytes'], rel
        inventory.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest, workers=sorted(assigned)))
    save(RUN/'input_manifest.json', dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                      source_manifest_sha256=sha(parent), files=inventory))
    packets = []
    for worker in workers:
        path = ROOT/f'cache/analysis035-212-worker{worker}.tar.gz'
        with tarfile.open(path,'w:gz',compresslevel=1) as tar:
            for row in inventory:
                if worker in row['workers']:
                    tar.add(ROOT/row['path'], arcname=row['path'])
            rel = (RUN/'input_manifest.json').relative_to(ROOT).as_posix()
            tar.add(RUN/'input_manifest.json', arcname=rel)
        packets.append(dict(worker=worker, path=path.relative_to(ROOT).as_posix(),
                            bytes=path.stat().st_size, sha256=sha(path)))
    runtime = json.loads((ROOT/'analyses/034-2026-11-01-eos-adam-continuations/runtime_bundle.json').read_text())
    assert sha(ROOT/runtime['path']) == runtime['sha256']
    save(HERE/'runtime_bundle.json', {k:runtime[k] for k in ['path','bytes','sha256']})
    save(HERE/'packets-212.json', dict(packets=packets))
    print(json.dumps(dict(jobs=len(jobs), workers=len(workers), packet_bytes=sum(p['bytes'] for p in packets))))


if __name__ == '__main__':
    main()
