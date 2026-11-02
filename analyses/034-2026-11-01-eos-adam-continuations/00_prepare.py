"""Freeze exact same 27 starts for a training-only cold-Adam LR calibration."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CAL = ROOT/'runs/209-2026-11-01-eos-adam-calibration'
PROD = ROOT/'runs/210-2026-11-01-eos-adam-long-lags'
OLD = ROOT/'runs/205-2026-11-01-eos-exact-lag10'
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import tensor_hash
from src.eos_tasks.task import model_for


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def packet(run):
    cfg=json.loads((run/'config.json').read_text())
    files={}
    def add(path,workers):files.setdefault(path,set()).update(workers)
    for j in cfg['jobs']:
        for field in ['checkpoint','anchor_gradient','data']:add(j[field],[j['worker']])
    workers=sorted({j['worker'] for j in cfg['jobs']})
    oldmanifest=json.loads((OLD/'input_manifest.json').read_text())
    shared=[r['path'] for r in oldmanifest['files'] if r['path'].startswith(('src/','cache/eos-multitask/reference/'))]
    shared+=['src/eos_tasks/lag_profile.py','src/eos_tasks/adam_profile.py',cfg['original_inputs'],
        (run/'config.json').relative_to(ROOT).as_posix(),(run/'01_worker.py').relative_to(ROOT).as_posix()]
    for path in shared:add(path,workers)
    inventory=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p),workers=sorted(w)) for p,w in sorted(files.items())]
    save(run/'input_manifest.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=inventory))
    packets=[]
    for worker in workers:
        path=ROOT/f'cache/analysis034-{cfg["run_id"]}-worker{worker}.tar.gz'
        with tarfile.open(path,'w:gz',compresslevel=1) as tar:
            for row in inventory:
                # Production reuses the verified calibration inputs on the same Pod.
                include='calibration' in cfg['phase'] or row['path'].startswith(run.relative_to(ROOT).as_posix()+'/')
                # Changed production placement needs those exact source files on its new worker.
                if not include:
                    old=json.loads((CAL/'input_manifest.json').read_text())
                    existing={x['path']:x['workers'] for x in old['files']}
                    include=worker not in existing.get(row['path'],[])
                if worker in row['workers'] and include:tar.add(ROOT/row['path'],arcname=row['path'])
            rel=(run/'input_manifest.json').relative_to(ROOT).as_posix()
            tar.add(run/'input_manifest.json',arcname=rel)
        packets.append(dict(worker=worker,path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    save(HERE/f'packets-{cfg["run_id"]}.json',dict(packets=packets))


def main():
    assert not (CAL/'config.json').exists(),'Calibration inputs already frozen'
    old=json.loads((OLD/'config.json').read_text())
    shared=json.loads((ROOT/old['original_inputs']).read_text())
    gd=json.loads((ROOT/'analyses/033-2026-11-01-eos-long-lags/results.json').read_text())['profiles']
    by_key={(r['parent_run_id'],r['anchor']):r for r in gd}
    jobs=[];provenance=[]
    mapping={}; count=0
    # Expensive independent states each get a worker; all cheap MLP states share one.
    for name in ['transformer','resnet']:
        for condition in old['conditions']:
            if condition['task']!=name:continue
            for anchor in old['anchors']:
                mapping[(condition['parent_run_id'],anchor)]=count;count+=1
    for condition in old['conditions']:
        rid,name=condition['parent_run_id'],condition['task']
        model=model_for(name,ROOT/old['task_cache'],shared['wikitext']['vocabulary_size'])
        base=OLD/f'artifacts/attempt-01/worker{condition["worker"]}'
        inv=json.loads((base/'inventory.json').read_text())
        receipt=json.loads((ROOT/f'analyses/032-2026-11-01-eos-exact-lag10/retrieval-worker{condition["worker"]}.json').read_text())
        assert sha(base/'inventory.json')==receipt['inventory_sha256']
        expected={r['path']:r for r in inv['files']}
        for anchor in old['anchors']:
            rel=f'{rid}/state-{anchor:06d}'; parent=base/rel
            sources=[]
            for filename in ['start.pt','result.json']:
                path=parent/filename;row=expected[rel+'/'+filename]
                assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
                sources.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
            result=json.loads((parent/'result.json').read_text())
            checkpoint=torch.load(parent/'start.pt',map_location='cpu',weights_only=False)
            assert checkpoint['step']==anchor
            model.load_state_dict(checkpoint['model'])
            assert tensor_hash(model.parameters())==result['starting_tensor_sha256']
            cache=ROOT/f'cache/analysis034-inputs/{rid}-{anchor}';cache.mkdir(parents=True,exist_ok=True)
            gradients=np.load(parent/'gradients.npy',mmap_mode='r')
            ref=ROOT/f'cache/analysis033-inputs/{rid}-{anchor}/anchor.npy'
            assert np.array_equal(np.load(ref),gradients[0])
            worker=18 if name=='mlp' else mapping[(rid,anchor)]
            jobs.append(dict(parent_run_id=rid,task=name,parent_gd_lr=condition['lr'],anchor=anchor,
                worker=worker,primary_eligible=result['primary_eligible'],parent_mode=result['parent_mode'],
                checkpoint=(parent/'start.pt').relative_to(ROOT).as_posix(),tensor_sha256=result['starting_tensor_sha256'],
                source_loss=result['rows'][0]['loss'],anchor_gradient=ref.relative_to(ROOT).as_posix(),
                data=condition['data'],reference_physical=condition['physical'],physical=512 if name=='transformer' else condition['physical'],
                rates=[1e-5,3e-5,1e-4,3e-4,1e-3],horizon=64,production_horizon=by_key[(rid,anchor)]['last_lag']))
            provenance.append(dict(parent_run_id=rid,anchor=anchor,sources=sources))
    cfg=dict(run_id='209',analysis='034',hypothesis='H026',phase='calibration',jobs=jobs,
        purpose='Calibrate one common cold-Adam LR per model across the same retained states; exact adjacent and anchored raw-gradient cosines',
        anchors=old['anchors'],objective=old['objective'],parameter_scope=old['parameter_scope'],
        precision=old['precision'],zero_denominator=old['zero_denominator'],validation='none; complete training objective only',
        optimizer=dict(name='Adam',betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False,
            state_initialization='fresh zero moments and step0 at every original anchor; discard parent SGD optimizer',schedule='constant',clipping='none'),
        task_cache=old['task_cache'],original_inputs=old['original_inputs'],
        inputs_manifest=CAL.relative_to(ROOT).as_posix()+'/input_manifest.json',
        stop_request=CAL.relative_to(ROOT).as_posix()+'/stop-request',gradient_relative_tolerance=1e-5,
        cosine_audit_tolerance=1e-10,recovery_every=32,
        calibration_rule='All5 rates for64 updates at all27 states. Exclude2 historically unqualified states from selection. Require finite coverage, peak loss<=4*initial+1e-6 and final16 mean<=1.05*initial+1e-6 at every primary state. Minimize equal-checkpoint mean log(final16 mean/initial) separately per model; ties choose smaller LR. Cosines and held-out loss never select LR. Conditional best on tested grid, not global optimum.',
        production_rule='Restart Adam fresh at identical original anchor, one selected common LR per model; match each Analysis033 GD terminal lag256/512/2048; no outcome-dependent stopping except finite/budget safeguard.',
        retention='All scalar products/CPU checks each lag; exact g0/g1/g2 and terminal adjacent pair per arm; terminal model/Adam/RNG with last gradient unapplied; source starts retained.',
        authorization='User explicitly authorizes Adam calibration then entire Runpod execution, additional funding and fastest practical parallel execution; prior preference to terminate all task Pods afterwards.',
        resource_limits=dict(max_total_usd=50,graceful_compute_stop_usd=47,hard_compute_stop_usd=49,
            max_lease_seconds=16200,max_concurrent_gpus=19,container_gb=30,pod_volume_gb=15),
        inference='One initialization; checkpoint-dependent forks, no seed error bars. Adam rates selected on short training losses; no SGD curvature identity for Adam, stability or generalization claim.')
    CAL.mkdir(parents=True,exist_ok=True);PROD.mkdir(parents=True,exist_ok=True)
    wrapper='from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[2]))\nfrom src.eos_tasks.adam_profile import main\nif __name__ == "__main__":main(Path(__file__).parent/"config.json")\n'
    (CAL/'01_worker.py').write_text(wrapper,encoding='utf-8',newline='\n')
    (PROD/'01_worker.py').write_text(wrapper,encoding='utf-8',newline='\n')
    save(CAL/'config.json',cfg);save(HERE/'source_provenance.json',dict(jobs=provenance))
    packet(CAL)
    print(json.dumps(dict(jobs=len(jobs),workers=19,primary=sum(j['primary_eligible'] for j in jobs),source_states='exact original anchor, not GD lag10')))


if __name__=='__main__':main()
