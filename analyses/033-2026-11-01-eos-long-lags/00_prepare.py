"""Verify Run205 evidence, extract streaming references and freeze Run206 packets."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
RUN = ROOT/'runs/206-2026-11-01-eos-long-lags'
OLD = ROOT/'runs/205-2026-11-01-eos-exact-lag10'
CACHE = ROOT/'cache/analysis033-inputs'
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import tensor_hash
from src.eos_tasks.task import model_for


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def main():
    assert not (RUN/'config.json').exists(), 'Inputs already frozen'
    CACHE.mkdir(exist_ok=True)
    old = json.loads((OLD/'config.json').read_text())
    original_manifest = json.loads((OLD/'input_manifest.json').read_text())
    source_hashes = {r['path']:r['sha256'] for r in original_manifest['files']}
    shared = json.loads((ROOT/old['original_inputs']).read_text())
    jobs = []
    files = {}
    provenance = []

    def add(path, workers):
        key = path.relative_to(ROOT).as_posix()
        files.setdefault(key, set()).update(workers)

    for worker in range(3):
        base = OLD/f'artifacts/attempt-01/worker{worker}'
        inv = json.loads((base/'inventory.json').read_text())
        receipt = json.loads((ROOT/f'analyses/032-2026-11-01-eos-exact-lag10/retrieval-worker{worker}.json').read_text())
        assert sha(base/'inventory.json') == receipt['inventory_sha256']
        expected = {r['path']:r for r in inv['files']}
        for condition in [c for c in old['conditions'] if c['worker'] == worker]:
            rid, name = condition['parent_run_id'], condition['task']
            model = model_for(name, ROOT/old['task_cache'], shared['wikitext']['vocabulary_size'])
            for anchor in old['anchors']:
                rel = f'{rid}/state-{anchor:06d}'
                paths = {n:base/rel/n for n in ['end.pt', 'gradients.npy', 'result.json']}
                for n, path in paths.items():
                    r = expected[rel+'/'+n]
                    assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256']
                result = json.loads(paths['result.json'].read_text())
                # Nine Transformer workers; three ResNet workers; all cheap MLP jobs on worker9.
                if name == 'transformer':
                    target_worker = (2-worker)*3 + (2-old['anchors'].index(anchor))
                elif name == 'resnet':
                    target_worker = 9+worker
                else:
                    target_worker = 9
                target = CACHE/f'{rid}-{anchor}'
                target.mkdir(exist_ok=True)
                gradients = np.load(paths['gradients.npy'], mmap_mode='r')
                np.save(target/'anchor.npy', gradients[0], allow_pickle=False)
                np.save(target/'boundary.npy', gradients[10], allow_pickle=False)
                shutil.copyfile(paths['end.pt'], target/'resume.pt')
                checkpoint = torch.load(target/'resume.pt', map_location='cpu', weights_only=False)
                assert checkpoint['step'] == anchor+10
                model.load_state_dict(checkpoint['model'])
                prefix = []
                for k, row in enumerate(result['rows']):
                    prefix.append(dict(lag=k, state=anchor+k, loss=row['loss'], norm=row['norm'],
                        cosine=row['anchor_cosine'], source='Run205 independently verified full vector'))
                job = dict(parent_run_id=rid, task=name, lr=condition['lr'], anchor=anchor,
                    worker=target_worker, seed=0, physical=condition['physical'], data=condition['data'],
                    checkpoint=(target/'resume.pt').relative_to(ROOT).as_posix(),
                    checkpoint_sha256=sha(target/'resume.pt'), resume_tensor_sha256=tensor_hash(model.parameters()),
                    anchor_gradient=(target/'anchor.npy').relative_to(ROOT).as_posix(),
                    boundary_gradient=(target/'boundary.npy').relative_to(ROOT).as_posix(),
                    source_window=paths['result.json'].relative_to(ROOT).as_posix(),
                    primary_eligible=result['primary_eligible'], parent_mode=result['parent_mode'],
                    initial_reversal=prefix[1]['cosine'] is not None and prefix[1]['cosine'] < 0,
                    prefix_rows=prefix)
                jobs.append(job)
                provenance.append(dict(parent_run_id=rid, anchor=anchor, primary_eligible=result['primary_eligible'],
                    sources=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                             for p in [paths['end.pt'], paths['result.json'], paths['gradients.npy']]],
                    extracted=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                               for p in [target/'resume.pt', target/'anchor.npy', target/'boundary.npy']]))
                for path in target.iterdir():
                    add(path, [target_worker])
                path = ROOT/condition['data']
                assert sha(path) == source_hashes[condition['data']]
                add(path, [target_worker])
            del model
    jobs.sort(key=lambda j:(j['worker'], {'mlp':0,'resnet':1,'transformer':2}[j['task']], j['parent_run_id'], j['anchor']))
    config = dict(run_id='206', analysis='033', hypothesis='H020',
        purpose='Extend Analysis032 exact anchored full-objective gradient profiles until oscillation damps or a declared bound',
        anchors=old['anchors'], stages=[256,512,1024,2048], jobs=jobs,
        metric=old['metric'], objective=old['objective'], optimizer=old['optimizer'],
        precision=old['precision'], parameter_scope=old['parameter_scope'], zero_denominator=old['zero_denominator'],
        task_cache=old['task_cache'], original_inputs=old['original_inputs'],
        inputs_manifest=RUN.relative_to(ROOT).as_posix()+'/input_manifest.json',
        stop_request=RUN.relative_to(ROOT).as_posix()+'/stop-request',
        gradient_relative_tolerance=1e-5, cosine_audit_tolerance=1e-10,
        transformer_compute_candidates=[64,128,256,512],
        compute_chunk_policy='One common Transformer chunk chosen by two complete-gradient timings per candidate at Run126 state3010; must match chunk64 within1e-5 relative gradient and1e-5 absolute loss, use<80% VRAM. Recheck chosen chunk at each of9 starts. FP32 accumulation partitions may differ; full objective, order, token denominator and model remain identical. ResNet fixed1000 BN groups never change.',
        envelope_width=64, damped_gap_threshold=.1,
        stopping_policy='All27 profiles reach256. Positive-lag1 controls stop there. Negative-lag1 profiles double to512/1024/2048 until the median half absolute odd/even gap of the last32 pairs is<=0.1. Keep all stages and report unequal coverage. Maximum horizon/budget may end undamped profiles; never guarantee decay.',
        recovery_every=128,
        retention='Exact GPU FP64 and independent CPU NumPy FP64 products at every new lag; raw full gradients at11 and the last odd/even pair; full reference/source history retained in Run205; current model/optimizer/RNG checkpoint every128 and at termination. No random projection or quantization.',
        qualification_policy=old['qualification_policy'], original_history_limit=3000,
        source_prefix='Reuse independently verified lags0..10 and exact Run205 end states; recompute the unapplied lag10 gradient before stepping. New continuations can differ in hardware and FP32 accumulation; not historical replay beyond anchor.',
        validation='none; complete training objective, no held-out claims',
        authorization='User asks far more lags until dampening is visual, permits compression if necessary, requests Runpod and fastest practical execution.',
        resource_limits=dict(workers=12, gpu_preference='RTX5090 Secure; RTX4090 Secure fallback if capacity unavailable',
            max_total_usd=30, graceful_compute_stop_usd=25, max_lease_seconds=14400,
            production_timeout_seconds=13600, container_gb=30, pod_volume_gb=8, network_volume=False),
        inference='One initialization, one anchored pair per lag; stopping coverage is adaptive and outcome dependent. Damping of this descriptive cosine envelope is not stability, generalization or an optimizer benefit.')
    save(RUN/'config.json', config)
    shared_files = [ROOT/old['original_inputs']]
    shared_files += [ROOT/n for n in ['src/__init__.py','src/cifar_eos/__init__.py','src/cifar_eos/core.py',
                                    'src/eos_tasks/__init__.py','src/eos_tasks/task.py','src/eos_tasks/production.py'] if (ROOT/n).exists()]
    shared_files += list((ROOT/'cache/eos-multitask/reference').glob('*.py'))
    shared_files += [RUN/'config.json', RUN/'01_worker.py']
    for path in shared_files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in source_hashes:
            assert sha(path) == source_hashes[rel], rel
        add(path, range(12))
    for n in ['source_inputs.json','source_cifar_inputs.json']:
        shutil.copyfile(OLD/n, RUN/n)
    save(RUN/'input_manifest.json', dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=n, bytes=(ROOT/n).stat().st_size, sha256=sha(ROOT/n), workers=sorted(ws)) for n,ws in sorted(files.items())]))
    save(HERE/'checkpoint_provenance.json', dict(windows=provenance))
    packets = []
    for worker in range(12):
        packet = ROOT/f'cache/analysis033-worker{worker}-inputs.tar.gz'
        chosen = [ROOT/n for n, ws in files.items() if worker in ws]+[RUN/'input_manifest.json']
        with tarfile.open(packet, 'w:gz', compresslevel=1) as tar:
            for path in chosen:
                tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        packets.append(dict(worker=worker,path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,
                            sha256=sha(packet),files=len(chosen)))
    save(HERE/'packets.json', dict(packets=packets,config_sha256=sha(RUN/'config.json')))
    print(json.dumps(dict(status='prepared', jobs=len(jobs), primary=sum(j['primary_eligible'] for j in jobs),
        packets=len(packets), total_packet_bytes=sum(p['bytes'] for p in packets))))


if __name__ == '__main__':
    main()
