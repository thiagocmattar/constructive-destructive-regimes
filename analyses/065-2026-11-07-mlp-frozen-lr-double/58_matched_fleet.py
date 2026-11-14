"""Task-owned H051 transport, bounded leases and verified Spark retention.

Only IDs created and recorded by this controller are managed. No credentials
are serialized; the task transfer key remains in ignored storage on Spark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
S=HERE/'scratch/h051'; S.mkdir(parents=True,exist_ok=True)
os.environ['PATH']=str(Path.home()/'.local/bin')+os.pathsep+os.environ['PATH']
spec=importlib.util.spec_from_file_location('h051transport',ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.SCRATCH=S
REMOTE='/workspace/h051'
SPARK='tcml-spark@spark-2277.local'
SPARKROOT='/home/tcml-spark/experiments/constructive-destructive-regimes/matched-interventions-2026-11-14'
PY='/opt/h051/bin/python'
OLD=json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
SPARKPY=OLD['python']; SPARKLIB=OLD['pythonpath']
IMAGE='pytorch/pytorch@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0'
IMAGE128='pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be'
RUNS={case:f'runs/{n}-2026-11-14-{case}-matched-interventions' for n,case in [(293,'mlp'),(294,'resnet'),(295,'transformer')]}
KEY=SPARKROOT+'/scratch/retention-key'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_bytes((json.dumps(value,indent=2)+'\n').encode());tmp.replace(path)


def emit(**d): print(json.dumps(dict(utc=x.stamp(),**d)),flush=True)
def record(p): dump(S/('lease-'+p['name']+'.json'),p)
def get(name): return json.loads((S/('lease-'+name+'.json')).read_text())


def spark(script,timeout=60):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',SPARK,'bash -s'],
        input=script.encode(),capture_output=True,timeout=timeout,creationflags=x.NO_WINDOW)
    if p.returncode: raise RuntimeError((p.stderr+p.stdout).decode(errors='replace')[-2000:])
    return p.stdout.decode()


def spark_copy(local,remote):
    p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',str(local),SPARK+':'+remote],
        capture_output=True,timeout=300,creationflags=x.NO_WINDOW)
    if p.returncode: raise RuntimeError(p.stderr.decode()[-1800:])


def start_spark(mechanics=False):
    case='mlp'; run=RUNS[case]; packet=json.loads((ROOT/run/'packet.json').read_text())
    spark('set -e\nmkdir -p '+shlex.quote(SPARKROOT)+'/scratch\n')
    spark_copy(ROOT/packet['path'],SPARKROOT+'/scratch/mlp-inputs.tar.gz')
    script=f'''set -e
cd {SPARKROOT}
echo '{packet['sha256']}  scratch/mlp-inputs.tar.gz' | sha256sum -c -
tar -xzf scratch/mlp-inputs.tar.gz
export PYTHONPATH={SPARKROOT}:{SPARKLIB} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
{SPARKPY} analyses/065-2026-11-07-mlp-frozen-lr-double/57_test_matched.py
'''
    if mechanics:
        run='analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h051/mechanics-mlp'
        script+=f'''{SPARKPY} - <<'PY'
import json,shutil
from pathlib import Path
p=Path({run!r});p.mkdir(parents=True,exist_ok=True)
d=json.loads(Path({(RUNS['mlp']+'/config.json')!r}).read_text());d.update(fork=16,end=18)
(p/'config.json').write_text(json.dumps(d))
shutil.copy2({(RUNS['mlp']+'/input_manifest.json')!r},p/'input_manifest.json')
PY
'''
    output=run+'/artifacts/attempt-01'
    arms=' '.join(a['id'] for a in json.loads((ROOT/RUNS['mlp']/'config.json').read_text())['arms'])
    script+=f'''{SPARKPY} -m src.eos_tasks.matched_interventions --run {run} --out {output}/shared --mode prefix
{SPARKPY} -m src.eos_tasks.matched_interventions --run {run} --out {output}/spark --shared {output}/shared --mode arms --arms {arms} {'--mechanics' if mechanics else ''}
'''
    tag='smoke' if mechanics else 'production'
    script='#!/bin/bash\n'+script+f'echo 0 > {SPARKROOT}/scratch/mlp-{tag}.exit\n'
    path=S/('mlp-'+tag+'.sh');path.write_bytes(script.encode())
    spark_copy(path,SPARKROOT+'/scratch/'+path.name)
    pid=spark(f'setsid bash {SPARKROOT}/scratch/{path.name} </dev/null >{SPARKROOT}/scratch/mlp-{tag}.log 2>&1 &\necho $!\n').strip()
    result=dict(pid=pid,root=SPARKROOT,python=SPARKPY,run=run,output=output,mechanics_only=mechanics,started_utc=x.stamp())
    dump(HERE/('matched_spark_'+tag+'_launch.json'),result);emit(event='spark_started',**result)


def create(name,case):
    assert name.startswith(case+'-') and case in ['resnet','transformer']
    assert not (S/('lease-'+name+'.json')).exists()
    live=[json.loads(f.read_text()) for f in S.glob('lease-*.json')]
    assert sum(not p.get('deleted_epoch') for p in live)<8
    gpu=os.environ.get('H051_GPU','NVIDIA GeForce RTX 4090' if case=='resnet' else 'NVIDIA GeForce RTX 5090')
    row=next(g for g in x.cli('gpu','list','--include-unavailable') if g['gpuId']==gpu)
    cloud=os.environ.get('H051_CLOUD','SECURE')
    price=row['securePricePerHr' if cloud=='SECURE' else 'communityPricePerHr']
    datacenter=os.environ.get('H051_DATACENTER')
    if datacenter:
        live_centers=x.cli('datacenter','list')
        assert any(c['id']==datacenter and any(g['gpuId']==gpu for g in c.get('gpuAvailability',[])) for c in live_centers)
    maximum_hourly=float(os.environ.get('H051_MAX_HOURLY',0.89 if case=='resnet' else 1.39))
    assert price is not None and price <= maximum_hourly
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    cuda=json.loads((ROOT/RUNS[case]/'config.json').read_text())['cuda_version']
    image=IMAGE128 if cuda=='12.8' else IMAGE
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    started=time.time()
    actual=x.create_wait('pod','create','--name','cdr-h051-'+name,'--image',image,'--gpu-id',gpu,
        '--gpu-count','1','--cloud-type',cloud,'--min-cuda-version',cuda,'--container-disk-in-gb','30',
        '--volume-in-gb','30','--ports','22/tcp','--ssh','--env',json.dumps({'PUBLIC_KEY':public}),
        '--docker-args','bash -lc '+shlex.quote(bootstrap),
        *(['--data-center-ids',datacenter] if datacenter else []),'--wait','--wait-timeout','20s')
    deadline=min([q['created_epoch'] for q in live]+[started])+8*3600
    p=dict(name=name,case=case,id=actual['id'],created_epoch=started,deadline_epoch=deadline,
        price_per_hour=actual.get('costPerHr') or price,storage_hourly_usd=60*.1/730,
        gpu=gpu,image=image,cuda=cuda,cloud=cloud,run=RUNS[case],phase='created',live_quote=row,
        requested_datacenter=datacenter)
    record(p);emit(event='created',name=name,id=p['id'],price=p['price_per_hour'])


def setup(name):
    p=get(name);assert not p.get('deleted_epoch')
    for _ in range(70):
        try:
            actual=x.cli('pod','get',p['id']);info=actual.get('ssh') or x.cli('ssh','info',p['id'])
            if info.get('ip') and info.get('port'):
                p.update(host=info['ip'],port=info['port']);record(p)
                x.ssh(p,'echo ready\n',timeout=20);break
        except Exception: pass
        time.sleep(6)
    else: raise RuntimeError('SSH unavailable; recorded lease remains under budget guard')
    x.ssh(p,f'''set -e
mkdir -p {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/h051
python3 -m pip --python /opt/h051 install --no-cache-dir numpy==2.5.0 scipy==1.18.1 threadpoolctl==3.6.0 >/workspace/setup.log 2>&1
{PY} -c "import torch; assert torch.__version__.split('+')[0]=='2.11.0' and torch.version.cuda=='{p.get('cuda','13.0')}'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
''',timeout=300)
    packet=json.loads((ROOT/p['run']/'packet.json').read_text())
    if os.environ.get('H051_TRANSFER_VIA_SPARK')=='1':
        authorize_spark(p)
        source=SPARKROOT+'/scratch/'+p['case']+'-inputs.tar.gz'
        command=['scp','-q','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile='+KEY+'.hosts','-i',KEY,'-P',str(p['port']),source,
            'root@'+p['host']+':/workspace/inputs.tar.gz']
        spark('set -e\n'+' '.join(shlex.quote(v) for v in command)+'\n',timeout=900)
    else:
        x.copy(p,ROOT/packet['path'],'/workspace/inputs.tar.gz')
    result=x.ssh(p,f'''set -e
echo '{packet['sha256']}  /workspace/inputs.tar.gz' | sha256sum -c -
tar -xzf /workspace/inputs.tar.gz -C {REMOTE}
cd {REMOTE}
PYTHONPATH={REMOTE} {PY} analyses/065-2026-11-07-mlp-frozen-lr-double/57_test_matched.py
''',timeout=240)
    p.update(phase='ready',packet_sha256=packet['sha256'],preflight=result,ready_utc=x.stamp());record(p)
    emit(event='ready',name=name)


def launch(name,mode,arms=None):
    p=get(name);assert not p.get('deleted_epoch') and p['phase'] in ['ready','prefix_ready']
    out=p['run']+'/artifacts/attempt-01/'+('shared' if mode=='prefix' else name)
    shared=p['run']+'/artifacts/attempt-01/shared'
    args=[PY,'-m','src.eos_tasks.matched_interventions','--run',p['run'],'--out',out,'--mode',mode]
    if mode=='arms': args+=['--shared',shared,'--arms',*arms]
    remaining=max(1,int(p['deadline_epoch']-time.time()-900))
    script=f'''#!/bin/bash
set -e
cd {REMOTE}
export PYTHONPATH={REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
set +e
timeout --signal=TERM --kill-after=60 {remaining} {' '.join(shlex.quote(v) for v in args)} > /workspace/{mode}.log 2>&1
code=$?
echo "$code" > /workspace/{mode}.exit
exit "$code"
'''
    local=S/(name+'-'+mode+'.sh');local.write_bytes(script.encode());x.copy(p,local,'/workspace/'+mode+'.sh')
    pid=x.ssh(p,f'setsid bash /workspace/{mode}.sh </dev/null >/workspace/{mode}-driver.log 2>&1 &\necho $!\n').strip()
    p.update(phase=mode,output=out,pid=pid,arms=arms,launched_utc=x.stamp());record(p)
    dump(ROOT/p['run']/(name+'_'+mode+'_launch.json'),p);emit(event='launched',name=name,mode=mode,arms=arms)


def status(name):
    p=get(name)
    if p.get('deleted_epoch'): return dict(name=name,status='deleted',verified=p.get('retrieval_verified'))
    if not p.get('host'): return dict(name=name,status=p['phase'])
    data=x.ssh(p,f'''python3 - <<'PY'
from pathlib import Path
import json,time
p=Path({(REMOTE+'/'+p.get('output',''))!r});d={{'name':{name!r},'phase':{p['phase']!r},'epoch':time.time()}}
for key in ['prefix','arms']:
 f=Path('/workspace')/(key+'.exit');d[key+'_exit']=f.read_text().strip() if f.exists() else None
for key in ['progress','manifest','ready','qualification']:
 f=p/(key+'.json')
 if f.exists():
  a=json.loads(f.read_text());d[key]=a if key in ['progress','ready'] else {{k:a[k] for k in ['status','error','eight_step_fingerprint','environment'] if k in a}}
for key in ['prefix','arms']:
 f=Path('/workspace')/(key+'.log')
 if f.exists():d[key+'_tail']=f.read_text()[-900:]
print(json.dumps(d))
PY
''',timeout=30)
    return json.loads(data)


def authorize_spark(p):
    public=spark(f'''set -e
mkdir -p {SPARKROOT}/scratch
if ! test -f {KEY}; then ssh-keygen -q -t ed25519 -N '' -f {KEY}; fi
chmod 600 {KEY}
cat {KEY}.pub
''').strip()
    x.ssh(p,'printf "%s\\n" '+shlex.quote(public)+' >> /root/.ssh/authorized_keys\n')


def copy_inventory(p,rel,inventory,tag):
    helper=HERE/'69_copy_matched_inventory.py';remote_helper=SPARKROOT+'/scratch/69_copy_matched_inventory.py'
    spark_copy(helper,remote_helper)
    request=dict(source_root=REMOTE+'/'+rel,destination_root=SPARKROOT+'/'+rel,
        host=p['host'],port=p['port'],key=KEY,files=inventory)
    local=S/(p['name']+'-'+tag+'-request.json');dump(local,request)
    remote_request=SPARKROOT+'/scratch/'+local.name;spark_copy(local,remote_request)
    result=spark(f'{SPARKPY} {shlex.quote(remote_helper)} {shlex.quote(remote_request)}\n',timeout=1800)
    return json.loads(result.strip().splitlines()[-1])


def retain(name,rel=None):
    p=get(name);authorize_spark(p);rel=rel or p['output']
    info=json.loads(x.ssh(p,f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil
root=Path({(REMOTE+'/'+rel)!r});files=[]
logs=root/'execution_logs';logs.mkdir(exist_ok=True)
for name in {(['prefix.log','setup.log'] if rel.endswith('/shared') else ['prefix.log','arms.log','setup.log'])!r}:
 p=Path('/workspace')/name
 if p.exists():shutil.copy2(p,logs/name)
for f in sorted(root.rglob('*')):
 if f.is_file() and f.name!='transfer_inventory.json':
  with f.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
  files.append(dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=digest))
(root/'transfer_inventory.json').write_text(json.dumps(dict(files=files)))
inventory=root/'transfer_inventory.json'
with inventory.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
files.append(dict(path='transfer_inventory.json',bytes=inventory.stat().st_size,sha256=digest))
print(json.dumps(dict(files=len(files),bytes=sum(f['bytes'] for f in files),inventory=files)))
PY
''',timeout=120))
    receipt=copy_inventory(p,rel,info['inventory'],'final')
    dump(ROOT/p['run']/(name+'_'+Path(rel).name+'_retention.json'),receipt)
    emit(event='retained_on_spark',name=name,**receipt)
    return receipt


def shared(name):
    p=get(name);assert status(name)['prefix_exit']=='0'
    receipt=retain(name,p['run']+'/artifacts/attempt-01/shared')
    p.update(phase='prefix_ready',shared_retained=receipt);record(p)


def distribute(name):
    p=get(name);authorize_spark(p)
    rel=p['run']+'/artifacts/attempt-01/shared'
    x.ssh(p,'mkdir -p '+shlex.quote(REMOTE+'/'+str(Path(rel).parent).replace('\\','/'))+'\n')
    command=['scp','-q','-r','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+KEY+'.hosts','-i',KEY,'-P',str(p['port']),SPARKROOT+'/'+rel,
        'root@'+p['host']+':'+REMOTE+'/'+str(Path(rel).parent).replace('\\','/')+'/']
    spark('set -e\n'+' '.join(shlex.quote(v) for v in command)+'\n',timeout=900)
    x.ssh(p,f'''{PY} - <<'PY'
from pathlib import Path
import json,hashlib
p=Path({(REMOTE+'/'+rel)!r})
for item in json.loads((p/'transfer_inventory.json').read_text())['files']:
 f=p/item['path']
 with f.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
 assert f.stat().st_size==item['bytes'] and digest==item['sha256']
print('Shared fork transfer verified')
PY
''',timeout=90)
    p.update(phase='prefix_ready');record(p);emit(event='fork_distributed',name=name)


def compact(name):
    p=get(name);rel=p['output']
    receipt=json.loads(x.ssh(p,f'''python3 - <<'PY'
import tarfile,hashlib,json
from pathlib import Path
root=Path({REMOTE!r});out=root/{rel!r};archive=Path('/workspace/compact.tar.gz')
files=[f for f in out.rglob('*') if f.is_file() and f.suffix in ['.json','.jsonl','.npz']]
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for f in files:t.add(f,arcname=f.relative_to(root).as_posix())
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=digest,files=len(files))))
PY
''',timeout=120))
    local=S/(name+'-compact.tar.gz');x.copy(p,local,'/workspace/compact.tar.gz',upload=False)
    assert local.stat().st_size==receipt['bytes'] and x.file_hash(local)==receipt['sha256']
    with tarfile.open(local) as t:t.extractall(ROOT,filter='data')
    dump(ROOT/p['run']/(name+'_compact.json'),receipt)


def collect(name):
    p=get(name);assert status(name)['arms_exit']=='0'
    verifier=HERE/'59_verify_matched.py';assert verifier.exists()
    rel=verifier.relative_to(ROOT).as_posix();x.copy(p,verifier,REMOTE+'/'+rel)
    x.ssh(p,f'cd {REMOTE}\nPYTHONPATH={REMOTE} {PY} {shlex.quote(rel)} --run {p["run"]} --out {p["output"]} --shared {p["run"]}/artifacts/attempt-01/shared\n',timeout=600)
    receipt=retain(name)
    compact(name)
    # The independent audit passed on this worker; raw evidence is now hashed on Spark.
    p.update(retrieval_verified=True,retention=receipt);record(p)
    x.cli('pod','delete',p['id']);p.update(deleted_epoch=time.time(),phase='deleted');record(p)
    dump(ROOT/p['run']/(name+'_closeout.json'),p);emit(event='verified_and_deleted',name=name)


def watch():
    futures={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        while not (S/'STOP_WATCHER').exists():
            now=time.time(); leases=[json.loads(f.read_text()) for f in S.glob('lease-*.json')]
            spent=sum((p.get('deleted_epoch',p.get('stopped_epoch',now))-p['created_epoch'])/3600*(p['price_per_hour']+p['storage_hourly_usd']) for p in leases)
            report=dict(utc=x.stamp(),estimated_spend_usd=spent,workers=[])
            for p in leases:
                if p.get('deleted_epoch'):continue
                name=p['name']
                if name in futures:
                    if futures[name].done():
                        try:futures[name].result()
                        except Exception as e:emit(event='collection_error',name=name,error=repr(e))
                        del futures[name]
                    else:
                        if (now>p['deadline_epoch']-300 or spent>48) and not p.get('stopped_epoch'):
                            x.cli('pod','stop',p['id']);p['stopped_epoch']=time.time();record(p)
                            emit(event='retrieval_budget_stop_volume_preserved',name=name,spend=spent)
                        continue
                try:
                    state=status(name);report['workers'].append(state)
                    if p['phase']=='arms' and state.get('arms_exit')=='0':
                        futures[name]=pool.submit(collect,name)
                    elif (now>p['deadline_epoch']-900 or spent>46) and not p.get('budget_stopped'):
                        # Preserve evidence: stop compute rather than deleting an unverified volume.
                        if p.get('pid'):x.ssh(p,'kill -TERM -- -'+str(p['pid'])+' 2>/dev/null || true\n')
                        p['budget_stopped']=True;record(p);emit(event='budget_guard',name=name,spend=spent)
                    if (now>p['deadline_epoch']-300 or spent>48) and not p.get('stopped_epoch'):
                        x.cli('pod','stop',p['id']);p['stopped_epoch']=time.time();record(p)
                        emit(event='compute_stopped_evidence_retained',name=name)
                except Exception as e:report['workers'].append(dict(name=name,error=repr(e)))
            dump(S/'monitor.json',report)
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['spark-smoke','spark-run','create','setup','prefix','shared','distribute','arms','status','collect','watch'])
    p.add_argument('name',nargs='?');p.add_argument('--case');p.add_argument('--arms',nargs='+');a=p.parse_args()
    if a.command=='spark-smoke':start_spark(True)
    elif a.command=='spark-run':start_spark(False)
    elif a.command=='create':create(a.name,a.case)
    elif a.command=='setup':setup(a.name)
    elif a.command=='prefix':launch(a.name,'prefix')
    elif a.command=='shared':shared(a.name)
    elif a.command=='distribute':distribute(a.name)
    elif a.command=='arms':launch(a.name,'arms',a.arms)
    elif a.command=='status':print(json.dumps(status(a.name),indent=2))
    elif a.command=='collect':collect(a.name)
    else:watch()
