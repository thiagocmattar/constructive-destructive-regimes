"""Task-owned Run269-271 cloud leases, detached jobs, retrieval and cost guard."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];S=HERE/'scratch/cross-architecture';S.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('cloud_transport',ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.SCRATCH=S
REMOTE='/workspace/cdr-transfer';LEDGER=HERE/'cross_architecture_leases.json'
RUNS={c:f'runs/{n}-2026-11-07-{c}-representation-interventions' for n,c in [(269,'mlp'),(270,'resnet'),(271,'transformer')]}
IMAGE='pytorch/pytorch@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0'


def dump(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n',newline='\n');q.replace(p)


def ledger():return json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(budget_usd=20,stop_compute_usd=18.5,maximum_live_gpus=3,pods=[])


def get(name):return next(p for p in ledger()['pods'] if p['name']==name)


def update(pod):
    d=ledger();d['pods']=[p for p in d['pods'] if p['name']!=pod['name']]+[pod];dump(LEDGER,d)


def log(**values):
    record=dict(utc=x.stamp());record.update(values);print(json.dumps(record),flush=True)


def start_watcher():
    if (S/'watcher.json').exists():return
    with (S/'watcher.log').open('xb') as out:
        p=subprocess.Popen([sys.executable,str(Path(__file__)),'watch'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,creationflags=x.NO_WINDOW)
    dump(S/'watcher.json',dict(pid=p.pid,started_utc=x.stamp()))


def create(name,gpu='NVIDIA GeForce RTX 4090'):
    d=ledger();assert name not in [p['name'] for p in d['pods']]
    active=[p for p in d['pods'] if not p.get('deleted_epoch')];assert len(active)<d['maximum_live_gpus']
    catalog=x.cli('gpu','list');g=next(g for g in catalog if g['gpuId']==gpu and g['available'])
    assert g['securePricePerHr']<=({'NVIDIA GeForce RTX 4090':.89,'NVIDIA GeForce RTX 5090':1.19}[gpu])
    dump(S/f'{name}-catalog.json',dict(utc=x.stamp(),gpu=g))
    now=time.time();committed=sum((p.get('deleted_epoch',p['deadline_epoch'])-p['created_epoch'])/3600*p['bounded_hourly_usd'] for p in d['pods'])
    hourly=float(g['securePricePerHr'])+(30+80)*.10/(30*24)+.01
    hours=min(6.,(d['stop_compute_usd']-committed)/hourly);assert hours>=2.,'Insufficient uncommitted budget for another lease'
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    start_watcher()
    actual=x.create_wait('pod','create','--name','cdr-h048-'+name,'--image',IMAGE,'--gpu-id',g['gpuId'],'--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
        '--container-disk-in-gb','30','--volume-in-gb','80','--ports','22/tcp','--ssh','--env',json.dumps({'PUBLIC_KEY':public}),
        '--docker-args','bash -lc '+shlex.quote(bootstrap),'--wait','--wait-timeout','20s')
    pod=dict(name=name,id=actual['id'],created_epoch=now,deadline_epoch=now+hours*3600,price_per_hour=actual.get('costPerHr',g['securePricePerHr']),
        bounded_hourly_usd=hourly,storage_gb=dict(container=30,volume=80),maximum_hours=hours,image=IMAGE,jobs=[])
    update(pod);dump(S/f'{name}-creation.json',actual);log(event='created',**pod)
    deadline=time.time()+600
    while time.time()<deadline:
        try:
            a=x.cli('pod','get',pod['id']);info=a.get('ssh') or x.cli('ssh','info',pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'],port=info['port']);update(pod);x.ssh(pod,'echo ready\n',timeout=20);break
        except Exception as e:log(event='waiting_for_ssh',error=str(e)[-300:])
        time.sleep(8)
    else:raise RuntimeError('Task pod failed SSH readiness; recorded ID requires cleanup')
    script=f'''set -e
mkdir -p {REMOTE} /workspace/transfer-control
python3 -m venv --system-site-packages --without-pip /opt/cdr-transfer
python3 -m pip --python /opt/cdr-transfer install --no-cache-dir numpy==2.5.0 scipy==1.18.1 threadpoolctl==3.6.0
/opt/cdr-transfer/bin/python -c "import torch; assert torch.__version__.split('+')[0]=='2.11.0'; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())"
'''
    f=S/f'{name}-setup.sh';f.write_text(script,newline='\n');x.copy(pod,f,'/workspace/transfer-control-setup.sh')
    x.ssh(pod,'setsid bash -c "bash /workspace/transfer-control-setup.sh > /workspace/setup.log 2>&1; echo \\$? > /workspace/setup.exit" </dev/null >/workspace/setup-driver.log 2>&1 &\necho $!\n')
    log(event='setup_detached',name=name)


def stage(name,case):
    pod=get(name);run=ROOT/RUNS[case];packet=json.loads((run/'packet.json').read_text());archive=ROOT/packet['path']
    assert x.file_hash(archive)==packet['sha256']
    result=x.ssh(pod,'if test -f /workspace/setup.exit; then cat /workspace/setup.exit; else echo WAIT; fi\n')
    assert result.strip()=='0',x.ssh(pod,'tail -c 2500 /workspace/setup.log\n')
    dest='/workspace/'+case+'-inputs.tar.gz';x.copy(pod,archive,dest)
    code=f'''/opt/cdr-transfer/bin/python - <<'PY'
import json,hashlib,tarfile,subprocess,os
from pathlib import Path
root=Path({REMOTE!r});p=Path({dest!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={packet['sha256']!r}
with tarfile.open(p) as t:
 for member in t.getmembers():
  target=root/member.name
  if target.exists() and target.is_file():
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(t.extractfile(member).read()).digest(),str(target)
 t.extractall(root,filter='data')
m=json.loads((root/{RUNS[case]!r}/'input_manifest.json').read_text())
for f in m['files']:
 p=root/f['path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
env=dict(os.environ,PYTHONPATH=str(root),CUBLAS_WORKSPACE_CONFIG=':4096:8')
p=subprocess.run(['/opt/cdr-transfer/bin/python',str(root/{(RUNS['mlp']+'/test_policy.py')!r})],cwd=root,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stderr
print(json.dumps(dict(status='passed',input_files=len(m['files']),tests=json.loads(p.stdout))))
PY
'''
    result=json.loads(x.ssh(pod,code,timeout=240));dump(run/f'{name}_preflight.json',result);log(event='staged',name=name,case=case,**result)


def launch(name,case,arms,mechanics=False,label=None,after=None):
    pod=get(name);label=label or case+('-mechanics' if mechanics else '-scientific');assert label not in [j['label'] for j in pod['jobs']]
    out=RUNS[case]+'/artifacts/'+name+'/'+label
    config=json.loads((ROOT/RUNS[case]/'config.json').read_text());worker=config.get('worker_script')
    entry=[RUNS[case]+'/'+worker] if worker else ['-m','src.eos_tasks.representation_continuation']
    args=['/opt/cdr-transfer/bin/python']+entry+['--run',RUNS[case],'--output',out,'--arms',arms]
    if mechanics:args+=['--mechanics']
    seconds=max(1,int(pod['deadline_epoch']-time.time()-900));assert seconds>600
    command=' '.join(shlex.quote(v) for v in args)
    prerequisite=''
    if after:
        assert after in [j['label'] for j in pod['jobs']]
        prerequisite=f'while ! test -f /workspace/{after}.exit; do sleep 30; done\nif test "$(cat /workspace/{after}.exit)" != 0; then echo 125 > /workspace/{label}.exit; exit 125; fi\n'
    script=f'''set -e
cd {REMOTE}
export PYTHONPATH={REMOTE} PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
set +e
{prerequisite}remaining=$(({int(pod['deadline_epoch'])} - $(date +%s) - 900))
timeout --signal=TERM --kill-after=30 "$remaining" {command} > /workspace/{label}.log 2>&1
code=$?
echo "$code" > /workspace/{label}.exit
exit "$code"
'''
    f=S/f'{name}-{label}.sh';f.write_text(script,newline='\n');dest='/workspace/'+label+'.sh';x.copy(pod,f,dest)
    pid=x.ssh(pod,'setsid bash '+shlex.quote(dest)+' </dev/null >/workspace/'+label+'-driver.log 2>&1 &\necho $!\n').strip()
    job=dict(label=label,case=case,run_dir=RUNS[case],arms=arms.split(','),output=out,mechanics=mechanics,after=after,pid=pid,started_utc=x.stamp())
    pod['jobs'].append(job);update(pod);dump(ROOT/RUNS[case]/f'{name}_{label}_launch.json',job);log(event='launched',name=name,**job)


def status(name):
    pod=get(name)
    code=f'''/opt/cdr-transfer/bin/python - <<'PY'
from pathlib import Path
import json,time
root=Path({REMOTE!r});jobs={pod['jobs']!r};r=[]
for j in jobs:
 out=root/j['output'];record=dict(label=j['label'])
 record['control_reuse_ready']=(out/'control_reuse.json').exists()
 e=Path('/workspace')/(j['label']+'.exit');record['exit']=e.read_text().strip() if e.exists() else None
 record['arms']={{p.parent.name:json.loads(p.read_text()) for p in out.glob('*/manifest.json')}}
 progress=list(out.glob('*/progress.json'))
 if progress:
  newest=max(progress,key=lambda p:p.stat().st_mtime);record['progress']=json.loads(newest.read_text())
 else:
  p=out/'progress.json';record['progress']=json.loads(p.read_text()) if p.exists() else None
 if record['exit'] not in [None,'0']:record['log']=Path('/workspace',j['label']+'.log').read_text()[-3000:]
 r.append(record)
print(json.dumps(dict(name={name!r},epoch=time.time(),jobs=r)))
PY
'''
    r=json.loads(x.ssh(pod,code));dump(S/f'{name}-status.json',r);print(json.dumps(r,indent=2));return r


def collect(name,label):
    pod=get(name);job=next(j for j in pod['jobs'] if j['label']==label)
    code=f'''/opt/cdr-transfer/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile,shutil
root=Path({REMOTE!r})/{job['output']!r};files=[]
root.mkdir(parents=True,exist_ok=True);logs=root/'infrastructure';logs.mkdir(exist_ok=True)
for name in [{job['label']+'.log'!r},{job['label']+'.exit'!r},{job['label']+'.sh'!r},{job['label']+'-audit.log'!r},'setup.log','setup.exit']:
 p=Path('/workspace')/name
 if p.exists():shutil.copy2(p,logs/name)
for p in sorted(root.rglob('*')):
 if p.is_file() and p!=root/'transfer_inventory.json':
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h))
receipt=root/'transfer_inventory.json';receipt.write_text(json.dumps(dict(files=files)))
archive=Path('/workspace')/{(name+'-'+label+'.tar.gz')!r}
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for f in files:t.add(root/f['path'],arcname=f['path'])
 t.add(receipt,arcname='transfer_inventory.json')
with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h,files=len(files),raw_bytes=sum(f['bytes'] for f in files))))
PY
'''
    run_dir=job.get('run_dir',RUNS[job['case']])
    receipt=json.loads(x.ssh(pod,code,timeout=900));target=ROOT/run_dir/'artifacts'/name/label;target.mkdir(parents=True,exist_ok=True)
    # Keep temporary compressed duplicates off the research volume when possible.
    archive_root=Path('R:/cdr-h048-transfers') if Path('R:/').is_dir() else target.parent
    archive_root.mkdir(parents=True,exist_ok=True);archive=archive_root/(name+'-'+label+'.tar.gz')
    x.copy(pod,archive,receipt['archive'],False)
    assert archive.stat().st_size==receipt['bytes'] and x.file_hash(archive)==receipt['sha256']
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    inventory=json.loads((target/'transfer_inventory.json').read_text())
    for f in inventory['files']:
        p=target/f['path'];assert p.stat().st_size==f['bytes'] and x.file_hash(p)==f['sha256'],str(p)
    assert archive.resolve().is_relative_to(archive_root.resolve())
    archive.unlink()  # Verified extracted files retain all evidence; avoid duplicate large archives.
    receipt.update(verified=True,local=target.relative_to(ROOT).as_posix(),utc=x.stamp(),duplicate_local_archive_removed=True)
    dump(ROOT/run_dir/f'{name}_{label}_retrieval.json',receipt)
    job['verified_retrieval']=True;update(pod);log(event='retrieved',name=name,label=label,**receipt)


def delete(name):
    pod=get(name);assert pod['jobs'] and all(j.get('verified_retrieval') for j in pod['jobs'])
    assert not pod.get('deleted_epoch');x.cli('pod','delete',pod['id']);pod['deleted_epoch']=time.time()
    pod['estimated_compute_usd']=(pod['deleted_epoch']-pod['created_epoch'])/3600*float(pod['price_per_hour']);update(pod)
    assert pod['id'] not in [p['id'] for p in x.cli('pod','list','--all')];log(event='deleted',name=name,id=pod['id'],estimated_compute_usd=pod['estimated_compute_usd'])


def watch():
    stopped=set()
    while not (S/'WATCHER_DONE').exists():
        d=ledger();now=time.time();spent=sum((p.get('deleted_epoch',now)-p['created_epoch'])/3600*p['bounded_hourly_usd'] for p in d['pods'])
        for pod in d['pods']:
            if pod.get('deleted_epoch') or pod['id'] in stopped:continue
            if now>=pod['deadline_epoch'] or spent>=d['stop_compute_usd']:
                try:
                    x.cli('pod','stop',pod['id']);stopped.add(pod['id']);log(event='budget_guard_stopped',id=pod['id'],bounded_spend=spent)
                except Exception as error:log(event='stop_retry',id=pod['id'],error=str(error))
        dump(S/'watcher-progress.json',dict(utc=x.stamp(),bounded_spend=spent,stopped=list(stopped)))
        time.sleep(30)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['create','stage','launch','status','collect','delete','watch']);a.add_argument('--name');a.add_argument('--case',choices=['resnet','transformer']);a.add_argument('--arms');a.add_argument('--label');a.add_argument('--mechanics',action='store_true');a.add_argument('--run');a.add_argument('--after');a.add_argument('--gpu',default='NVIDIA GeForce RTX 4090');q=a.parse_args()
    if q.run:
        target=(ROOT/q.run).resolve();assert target.is_relative_to(ROOT/'runs') and (target/'config.json').exists()
        RUNS[q.case]=target.relative_to(ROOT).as_posix()
    if q.action=='create':create(q.name,q.gpu)
    elif q.action=='stage':stage(q.name,q.case)
    elif q.action=='launch':launch(q.name,q.case,q.arms,q.mechanics,q.label,q.after)
    elif q.action=='status':status(q.name)
    elif q.action=='collect':collect(q.name,q.label)
    elif q.action=='delete':delete(q.name)
    else:watch()
