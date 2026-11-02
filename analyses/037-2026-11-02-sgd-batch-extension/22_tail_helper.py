"""Qualify, execute and verify two unchanged tail forks, then hand them to Run214."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import torch
torch.set_num_threads(4)

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
HELPER='tvo0a8tlrniihp';PARENT='92xz66q1cvuby0'
NAMES=[f'state-2000-b20-rep{rep:02d}' for rep in [6,7]]


def pods():
    state=c.lease()
    return [next(p for p in state['pods'] if p['id']==key) for key in [HELPER,PARENT]]


def start():
    pod,parent=pods();assert not pod.get('helper_pid')
    status=c.status(pod);assert status.get('calibration_exit')=='0'
    c.copy(pod,HERE/'02_verify_results.py',c.CONTROL+'/verify.py')
    result=c.ssh(pod,f'''set -e
/opt/cdr214-venv/bin/python - <<'PY'
import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,{c.REMOTE!r})
s=importlib.util.spec_from_file_location('v',{(c.CONTROL+'/verify.py')!r});v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
v.ROOT=Path('/workspace');v.RUN=Path({(c.REMOTE+'/'+c.REL)!r})
r=v.verify_worker(Path({(c.CONTROL+'/short-worker')!r}),mechanics=True,input_manifest=v.RUN/'input_manifest.json')
print(json.dumps(dict(status='passed',mechanics_only=True,verified_forks=len(r))))
PY
''',timeout=120)
    qualification=json.loads(result);qualification.update(utc=c.stamp(),pod_id=HELPER,role=pod['role'])
    c.save(HERE/'tail-helper-qualification.json',qualification)
    c.retrieve(pod,'calibration')
    reference=c.RUN/'artifacts/attempt-04/resnet-seed2/reference/state-2000.pt'
    c.ssh(pod,f'mkdir -p {c.outpath(pod)}/reference\n')
    target=c.outpath(pod)+'/reference/state-2000.pt';c.copy(pod,reference,target)
    assert c.ssh(pod,f'sha256sum {target}\n').split()[0]==c.file_hash(reference)
    c.copy(pod,HERE/'21_tail_worker.py',c.CONTROL+'/tail-worker.py')
    deadline=min(c.lease()['scientific_deadline_epoch'],time.time()+1200)
    body=f'''set -euo pipefail
trap 'echo $? > {c.CONTROL}/helper.exit' EXIT
cd {c.REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/opt/cdr214-venv/bin/python -u {c.CONTROL}/tail-worker.py --root {c.REMOTE} --output {c.outpath(pod)} --deadline-epoch {deadline}
'''
    local=c.SCRATCH/'tail-helper.sh';local.write_text(body,encoding='utf-8',newline='\n');c.copy(pod,local,c.CONTROL+'/helper.sh')
    pid=c.ssh(pod,f'set -e\ntest ! -e {c.CONTROL}/helper.pid\nsetsid bash {c.CONTROL}/helper.sh > {c.CONTROL}/helper.log 2>&1 < /dev/null &\necho $! | tee {c.CONTROL}/helper.pid\n').strip()
    pod.update(helper_pid=pid,helper_started_epoch=time.time())
    state=c.lease();next(p for p in state['pods'] if p['id']==HELPER).update(pod);c.save(c.SCRATCH/'lease.json',state)
    c.save(HERE/'tail-helper-launch.json',dict(utc=c.stamp(),pod_id=HELPER,parent_pod=PARENT,
        jobs=NAMES,deadline_epoch=deadline,config_sha256=c.file_hash(c.RUN/'config.json'),
        input_manifest_sha256=c.file_hash(c.RUN/'input_manifest.json'),
        reference_sha256=c.file_hash(reference),worker_script_sha256=c.file_hash(HERE/'21_tail_worker.py')))
    print(json.dumps(dict(status='launched',pod_id=HELPER,jobs=NAMES)),flush=True)


def finish():
    pod,parent=pods();receipt_path=HERE/f'retrieval-{pod["name"]}-production.json'
    if receipt_path.exists():folder=c.ROOT/c.read(receipt_path)['path']
    else:
        assert c.ssh(pod,f'cat {c.CONTROL}/helper.exit\n').strip()=='0'
        folder=c.retrieve(pod,'production')
    manifest=c.read(folder/'worker_manifest.json');assert manifest['status']=='complete' and manifest['complete_jobs']==2
    assert [j['name'] for j in manifest['identity']['jobs']]==NAMES
    inventory=c.read(folder/'worker_inventory.json')
    assert set(inventory)=={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='worker_inventory.json'}
    for name,entry in inventory.items():
        path=folder/name;assert path.resolve().is_relative_to(folder.resolve())
        assert path.stat().st_size==entry['bytes'] and c.file_hash(path)==entry['sha256']
    origin=torch.load(folder/'reference/state-2000.pt',map_location='cpu',weights_only=False)
    for name,rep in zip(NAMES,[6,7]):
        target=folder/name;summary=c.read(target/'summary.json');identity=summary['identity']
        assert summary['status']=='complete' and summary['updates']==200 and summary['valid_cosines']==199
        assert (identity['task'],identity['seed'],identity['state'],identity['batch'],identity['repetition'])==('resnet',2,2000,20,rep)
        assert identity['config_sha256']==c.file_hash(c.RUN/'config.json')
        assert identity['input_manifest_sha256']==c.file_hash(c.RUN/'input_manifest.json')
        assert identity['source']['sha256']==c.file_hash(folder/'reference/state-2000.pt')
        rows=[json.loads(line) for line in (target/'metrics.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==list(range(1,201)) and rows[0]['cosine'] is None
        np.testing.assert_allclose(np.mean([r['cosine'] for r in rows[1:]]),summary['mean_rho'],rtol=1e-12)
        np.testing.assert_allclose(rows[-1]['B']/sum(r['norm'] for r in rows),summary['R'],rtol=1e-12)
        end=torch.load(target/'final.pt',map_location='cpu',weights_only=False);assert end['state']==2200
        distance=sum(float((end['model'][k].double()-x.double()).square().sum()) for k,x in origin['model'].items() if x.is_floating_point())**.5
        np.testing.assert_allclose(distance,summary['absolute_displacement'],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(distance/sum(r['actual']['norm'] for r in rows),summary['actual_R'],rtol=1e-8,atol=1e-9)
        rng=np.random.default_rng(np.random.SeedSequence([20261102,1,2,2000,rep]))
        for row in np.fromfile(target/'indices.u32',dtype='<u4').reshape(200,20):np.testing.assert_array_equal(row,rng.permutation(50)[:20])
    receipt=c.read(receipt_path);receipt.update(scientific_status='complete',role=pod['role'],inventory_sha256=c.file_hash(folder/'worker_inventory.json'))
    c.save(HERE/('stream-retrieval-'+pod['name']+'.json'),receipt)
    c.save(HERE/('worker-acceptance-'+pod['name']+'.json'),dict(status='passed',utc=c.stamp(),role=pod['role'],
        pod_id=HELPER,task='resnet',seed=2,verified_forks=2,inventory_sha256=receipt['inventory_sha256'],verifier_sha256=c.file_hash(Path(__file__))))
    if not pod.get('deleted_epoch'):
        c.delete(pod);state=c.lease();next(p for p in state['pods'] if p['id']==HELPER).update(pod);c.save(c.SCRATCH/'lease.json',state)
    archive=c.SCRATCH/'tail-handoff.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for name in NAMES:bundle.add(folder/name,arcname=name)
    remote=c.CONTROL+'/tail-handoff.tar.gz';c.copy(parent,archive,remote)
    acknowledgement=c.ssh(parent,f'''set -e
echo '{c.file_hash(archive)}  {remote}' | sha256sum -c - > /dev/null
python3 - <<'PY'
from pathlib import Path
import json,tarfile,os,hashlib
root=Path({c.outpath(parent)!r});staging=Path({(c.CONTROL+'/tail-incoming')!r});staging.mkdir(exist_ok=True)
with tarfile.open({remote!r}) as bundle:bundle.extractall(staging,filter='data')
jobs=json.loads((root/'worker_manifest.json').read_text())['identity']['jobs'];current=json.loads((root/'worker_progress.json').read_text())['job_index'];accepted=[]
for name in {NAMES!r}:
 target=root/name;source=staging/name;index=next(i for i,j in enumerate(jobs) if j['name']==name)
 if (target/'summary.json').exists():
  with (target/'summary.json').open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
  with (source/'summary.json').open('rb') as stream:incoming=hashlib.file_digest(stream,'sha256').hexdigest()
  assert existing==incoming,name
  accepted.append(dict(name=name,summary_sha256=existing,already_imported=True));continue
 assert index-current>=3 and not target.exists(),name
 for relative,entry in json.loads((source/'inventory.json').read_text()).items():
  path=source/relative
  with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  assert path.stat().st_size==entry['bytes'] and digest==entry['sha256']
 os.rename(source,target)
 with (target/'summary.json').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 accepted.append(dict(name=name,summary_sha256=digest))
print(json.dumps(dict(accepted=accepted,parent_job_index=current)))
PY
''',timeout=120)
    ack=json.loads(acknowledgement)
    for item in ack['accepted']:assert item['summary_sha256']==c.file_hash(folder/item['name']/'summary.json')
    c.save(HERE/'tail-helper-handoff.json',dict(status='passed',utc=c.stamp(),source_pod=HELPER,
        parent_pod=PARENT,path=folder.relative_to(c.ROOT).as_posix(),**ack))
    # The frozen queue can skip imported arms immediately. Hold that queue only
    # if its remaining verified small-batch handoffs have not arrived yet.
    c.copy(parent,HERE/'23_queue_barrier.py',c.CONTROL+'/tail-barrier.py')
    command=f'/opt/cdr214-venv/bin/python -u {c.CONTROL}/tail-barrier.py --worker {c.outpath(parent)} --receipt {c.CONTROL}/tail-barrier.json --deadline-epoch {c.lease()["scientific_deadline_epoch"]}'
    pid=c.ssh(parent,f'set -e\nif test -e {c.CONTROL}/tail-barrier.pid; then cat {c.CONTROL}/tail-barrier.pid; else\nsetsid {command} > {c.CONTROL}/tail-barrier.log 2>&1 < /dev/null &\necho $! | tee {c.CONTROL}/tail-barrier.pid\nfi\n').strip()
    c.save(HERE/'tail-barrier-launch.json',dict(utc=c.stamp(),pod_id=PARENT,pid=pid,script_sha256=c.file_hash(HERE/'23_queue_barrier.py')))
    print(json.dumps(dict(status='passed',verified_forks=2,helper_deleted=True,handed_off=ack)),flush=True)


if __name__=='__main__':{'start':start,'finish':finish}[sys.argv[1]]()
