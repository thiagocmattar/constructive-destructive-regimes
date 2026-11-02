"""Task-owned two-phase Runpod batch lifecycle with detached work and hash gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
CAL=ROOT/'runs/209-2026-11-01-eos-adam-calibration'
PROD=ROOT/'runs/210-2026-11-01-eos-adam-long-lags'
SCRATCH=HERE/'scratch'
REMOTE='/workspace/constructive'
PREFIX='cdr-analysis034-'
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)
sys.path.insert(0,str(ROOT))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def cli(*args,check=True,timeout=75):
    p=subprocess.run(['runpodctl',*args],capture_output=True,text=True,timeout=timeout,creationflags=NO_WINDOW)
    if p.returncode:
        if check:raise RuntimeError(p.stderr[:2500])
        return None
    return json.loads(p.stdout)


def lease():return json.loads((SCRATCH/'lease.json').read_text())


def transport(pod,scp=False):
    return ['scp' if scp else 'ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/analysis034-known-hosts'),
        '-i',str(Path.home()/'.ssh/id_ed25519_runpod'),'-P' if scp else '-p',str(pod['port'])]


def ssh(pod,script,timeout=90):
    p=subprocess.run(transport(pod)+['root@'+pod['host'],'bash -s'],input=script.encode(),capture_output=True,
        timeout=timeout,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:]+p.stdout.decode(errors='replace')[-2000:])
    return p.stdout.decode()


def copy(pod,source,destination,upload=False):
    remote='root@'+pod['host']+':'
    args=[str(source),remote+destination] if upload else [remote+source,str(destination)]
    p=subprocess.run(transport(pod,True)+['-q',*args],capture_output=True,timeout=1800,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])


def control(run_id):return f'/workspace/analysis034-control-{run_id}'


def runpath(run_id):return CAL if run_id=='209' else PROD


def output(pod,run_id):
    return f'{REMOTE}/{runpath(run_id).relative_to(ROOT).as_posix()}/artifacts/attempt-{pod["attempt"]:02d}/worker{pod["worker"]}'


def receipt_path(pod,run_id):return HERE/f'retrieval-{run_id}-worker{pod["worker"]}-attempt{pod["attempt"]}.json'


def capture_calibration(pod):
    """Small terminal scalar record permits selection while full evidence transfers."""
    path=HERE/f'calibration-scalar-worker{pod["worker"]}.json'
    if path.exists():return
    state=status(pod,'209');assert state.get('worker_exit')=='0'
    out=output(pod,'209')
    script=f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib
p=Path('{out}');rows=[]
for q in sorted(p.glob('*-*/result.json')):
 b=q.read_bytes();rows.append(dict(path=q.relative_to(p).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b)))
print(json.dumps(dict(manifest=json.loads((p/'manifest.json').read_text()),results=rows)))
PY
'''
    result=json.loads(ssh(pod,script))
    assert result['manifest']['status']=='complete' and result['manifest']['config_sha256']==sha(CAL/'config.json')
    result.update(pod_id=pod['id'],worker=pod['worker'],attempt=pod['attempt'],epoch=time.time())
    save(path,result)


def allocate(worker,force_gpu=None):
    assert not (HERE/'closeout.json').exists(),'Closed cohort'
    l=lease();limits=l['limits']
    assert sum(not p.get('deleted_epoch') for p in l['pods'])<limits['max_concurrent_gpus']
    gpus=cli('gpu','list','--include-unavailable')
    preferences=[force_gpu] if force_gpu else (['RTX 5090','RTX 4090'] if worker<9 else ['RTX 4090','RTX 5090'])
    templates=cli('template','list','--type','official')
    image=next(t['imageName'] for t in templates if t['id']=='runpod-torch-v280')
    attempt=1+sum(p['worker']==worker for p in l['pods'])
    name=f'{PREFIX}worker{worker}-attempt{attempt}'
    assert not any(p['name']==name for p in cli('pod','list')),'Ambiguous existing create'
    for gpu_name in preferences:
        gpu=next(g for g in gpus if g['displayName']==gpu_name)
        assert gpu['securePricePerHr']<=({'RTX 4090':.89,'RTX 5090':1.19}[gpu_name])
        if not gpu['available']:continue
        try:
            result=cli('pod','create','--name',name,'--image',image,'--gpu-id',gpu['gpuId'],
                '--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
                '--container-disk-in-gb','30','--volume-in-gb','15','--ports','22/tcp','--ssh')
            break
        except RuntimeError as e:
            if not any(s in str(e).lower() for s in ['no available','insufficient','capacity','no instances','not have the resources']):raise
            save(SCRATCH/f'capacity-worker{worker}-{gpu_name.replace(" ","")}.json',dict(error=str(e)))
    else:return None
    pod=dict(id=result['id'],name=name,worker=worker,attempt=attempt,created_epoch=time.time(),
        gpu=gpu_name,price_per_gpu_hour=gpu['securePricePerHr'],phase='209')
    save(HERE/f'create-worker{worker}-attempt{attempt}.json',result)
    l['pods'].append(pod);save(SCRATCH/'lease.json',l)
    print(json.dumps(dict(created=pod)),flush=True)
    return pod


def connect(pod):
    info=cli('ssh','info',pod['id'],check=False)
    if not info or not info.get('ip'):
        # Current CLI ssh-info can lag behind pod-get's mapped public TCP route.
        state=cli('pod','get',pod['id'])
        info=state.get('ssh')
    if info and info.get('ip') and info.get('port'):
        pod.update(host=info['ip'],port=info['port'])
        ssh(pod,'echo transport-ready\n',timeout=25)
        return True
    return False


def setup(pod,run_id):
    marker=HERE/f'setup-{run_id}-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
    if marker.exists():return
    packet=next(p for p in json.loads((HERE/f'packets-{run_id}.json').read_text())['packets'] if p['worker']==pod['worker'])
    ctrl=control(run_id);ssh(pod,f'mkdir -p {ctrl}\n')
    copy(pod,ROOT/packet['path'],f'/tmp/analysis034-{run_id}-inputs.tar.gz',upload=True)
    install=''
    if run_id=='209':
        wheels=ROOT/'cache/analysis033-wheels.tar'
        copy(pod,wheels,'/tmp/analysis034-wheels.tar',upload=True)
        install=f'''printf '%s  %s\\n' '{sha(wheels)}' /tmp/analysis034-wheels.tar | sha256sum -c -
mkdir -p /tmp/analysis034-wheels
tar -xf /tmp/analysis034-wheels.tar -C /tmp/analysis034-wheels
python3 -m venv --system-site-packages /opt/analysis034-venv
/opt/analysis034-venv/bin/python -m pip install --no-index --find-links /tmp/analysis034-wheels cuda-bindings==13.4.4 cuda-pathfinder==1.8.4 numpy==2.5.0 scipy==1.18.1
/opt/analysis034-venv/bin/python -m pip install --no-cache-dir --timeout 45 --retries 2 --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
'''
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > {ctrl}/setup.exit' EXIT
printf '%s  %s\\n' '{packet['sha256']}' /tmp/analysis034-{run_id}-inputs.tar.gz | sha256sum -c -
mkdir -p {REMOTE}
tar -xzf /tmp/analysis034-{run_id}-inputs.tar.gz -C {REMOTE}
{install}
/opt/analysis034-venv/bin/python -c 'import torch,numpy,scipy; assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name())'
'''
    path=SCRATCH/f'setup-{run_id}-{pod["worker"]}-{pod["attempt"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
    copy(pod,path,ctrl+'/setup.sh',upload=True)
    pid=ssh(pod,f'setsid bash {ctrl}/setup.sh > {ctrl}/setup.log 2>&1 < /dev/null &\necho $!\n').strip()
    save(marker,dict(epoch=time.time(),pid=pid,packet_sha256=packet['sha256']))


def status(pod,run_id):
    if pod.get('deleted_epoch'):return dict(deleted=True)
    ctrl=control(run_id);out=output(pod,run_id)
    script=f'''python3 - <<'PY'
from pathlib import Path
import json,time,shutil
c=Path('{ctrl}');o=Path('{out}')
r={{'worker':{pod['worker']},'run_id':'{run_id}','epoch':time.time(),'free_gb':shutil.disk_usage('/workspace').free/1e9}}
for n in ['setup','worker']:
 p=c/(n+'.exit')
 if p.exists():r[n+'_exit']=p.read_text().strip()
for n in ['progress','manifest']:
 p=o/(n+'.json')
 if p.exists():r[n]=json.loads(p.read_text())
for n in ['setup','worker']:
 p=c/(n+'.log')
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-1300));r[n+'_tail']=f.read().decode(errors='replace')
print(json.dumps(r))
PY
'''
    return json.loads(ssh(pod,script,timeout=40))


def launch(pod,run_id):
    marker=HERE/f'launch-{run_id}-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
    if marker.exists():return
    ctrl=control(run_id);out=output(pod,run_id);rel=runpath(run_id).relative_to(ROOT).as_posix()
    script=f'''#!/bin/bash
set -uo pipefail
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM 15000 /opt/analysis034-venv/bin/python {rel}/01_worker.py --worker {pod['worker']} --output {out} > {ctrl}/worker.log 2>&1
echo $? > {ctrl}/worker.exit
'''
    path=SCRATCH/f'launch-{run_id}-{pod["worker"]}-{pod["attempt"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
    copy(pod,path,ctrl+'/worker.sh',upload=True)
    pid=ssh(pod,f'test ! -e {shlex.quote(out)}\nsetsid bash {ctrl}/worker.sh > {ctrl}/driver.log 2>&1 < /dev/null &\necho $!\n').strip()
    save(marker,dict(epoch=time.time(),pid=pid,output=out,config_sha256=sha(runpath(run_id)/'config.json')))


def retrieve(pod,run_id,failed=False):
    receipt=receipt_path(pod,run_id)
    if receipt.exists():return json.loads(receipt.read_text())
    s=status(pod,run_id)
    assert s.get('worker_exit') is not None or failed,'Work still active'
    out=output(pod,run_id);ctrl=control(run_id)
    script=f'''set -e
mkdir -p {out}/control
cp -a {ctrl}/. {out}/control/
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('{out}')
if not (p/'manifest.json').exists():(p/'manifest.json').write_text(json.dumps(dict(status='infrastructure_failed')))
files=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='inventory.json':
  with f.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
  files.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h))
(p/'inventory.json').write_text(json.dumps(dict(files=files),indent=2)+'\\n')
PY
tar -I 'gzip -1' -cf /tmp/analysis034-{run_id}-outputs.tar.gz -C {out} .
sha256sum /tmp/analysis034-{run_id}-outputs.tar.gz
'''
    digest=ssh(pod,script,timeout=900).split()[0]
    archive=ROOT/f'cache/analysis034-{run_id}-worker{pod["worker"]}-attempt{pod["attempt"]}-outputs.tar.gz'
    copy(pod,f'/tmp/analysis034-{run_id}-outputs.tar.gz',archive)
    assert sha(archive)==digest,'Transfer archive SHA mismatch'
    dest=runpath(run_id)/f'artifacts/attempt-{pod["attempt"]:02d}/worker{pod["worker"]}';dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:t.extractall(dest,filter='data')
    inv=json.loads((dest/'inventory.json').read_text());total=0
    for item in inv['files']:
        path=(dest/item['path']).resolve()
        assert path.is_relative_to(dest.resolve())
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
        total+=item['bytes']
    m=json.loads((dest/'manifest.json').read_text())
    assert m['status']=='complete' or failed,m
    result=dict(pod_id=pod['id'],worker=pod['worker'],attempt=pod['attempt'],run_id=run_id,
        verified=True,scientific_status=m['status'],artifact_root=dest.relative_to(ROOT).as_posix(),files=len(inv['files']),
        bytes=total,inventory_sha256=sha(dest/'inventory.json'),archive_sha256=digest,epoch=time.time())
    save(receipt,result);return result


def delete(pod,required_runs):
    for run_id in required_runs:
        receipt=json.loads(receipt_path(pod,run_id).read_text())
        assert receipt['verified'] and receipt['pod_id']==pod['id']
    actual=cli('pod','get',pod['id'])
    assert actual['name']==pod['name'] and actual['name'].startswith(PREFIX)
    save(HERE/f'final-pod-worker{pod["worker"]}-attempt{pod["attempt"]}.json',actual)
    cli('pod','delete',pod['id'])
    probe=subprocess.run(['runpodctl','pod','get',pod['id']],capture_output=True,text=True,creationflags=NO_WINDOW,timeout=45)
    assert probe.returncode!=0 and json.loads(probe.stderr.splitlines()[0]).get('code')=='not_found'
    pod['deleted_epoch']=time.time()


def production():
    if (PROD/'config.json').exists():return
    from src.eos_tasks.adam_profile import select_rate
    cfg=json.loads((CAL/'config.json').read_text())
    results=[]
    parallel=json.loads((HERE/'parallel_calibration_ready.json').read_text()) if (HERE/'parallel_calibration_plan.json').exists() else None
    for worker in range(19):
        if parallel and worker in [1,2]:continue
        scalar=HERE/f'calibration-scalar-worker{worker}.json'
        if scalar.exists():
            results.extend(r['value'] for r in json.loads(scalar.read_text())['results']);continue
        r=next(r for p in reversed(lease()['pods']) if p['worker']==worker
            for r in [receipt_path(p,'209')] if r.exists() and json.loads(r.read_text())['scientific_status']=='complete')
        receipt=json.loads(r.read_text())
        root=ROOT/receipt['artifact_root']
        results.extend(json.loads(p.read_text()) for p in root.glob('*-*/result.json'))
    if parallel:results.extend(parallel['canonical_results'])
    assert len(results)==27
    decisions={name:select_rate([r for r in results if r['task']==name],cfg['jobs'][0]['rates'])
        for name in ['mlp','resnet','transformer']}
    save(HERE/'calibration_decision.json',dict(decisions=decisions,calibration_results=results))
    assert all(d['selected_lr'] is not None for d in decisions.values()),'No eligible rate in declared grid'
    cfg.update(run_id='210',phase='production',inputs_manifest=PROD.relative_to(ROOT).as_posix()+'/input_manifest.json',
        stop_request=PROD.relative_to(ROOT).as_posix()+'/stop-request',recovery_every=128,
        purpose='Matched-horizon cold-Adam continuation from the same27 retained states, at the calibrated common rate per model; exact anchored and adjacent raw-gradient cosines',
        selection_sha256=sha(HERE/'calibration_decision.json'))
    for job in cfg['jobs']:
        job['rates']=[decisions[job['task']]['selected_lr']]
        job['horizon']=job['production_horizon']
        if parallel and job['parent_run_id']=='121' and job['anchor'] in [2000,3000]:
            job['worker']={2000:9,3000:11}[job['anchor']]
    if parallel:cfg['parallel_calibration_config_sha256']=parallel['config_sha256']
    save(PROD/'config.json',cfg)
    spec=importlib.util.spec_from_file_location('prepare034',HERE/'00_prepare.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.packet(PROD)
    print(json.dumps(dict(selected={k:v['selected_lr'] for k,v in decisions.items()})),flush=True)


def tick(pod):
    run_id=pod['phase']
    try:
        if not pod.get('host'):
            if not connect(pod):return pod
        setup(pod,run_id)
        s=status(pod,run_id)
        if s.get('setup_exit')=='0':launch(pod,run_id)
        if s.get('worker_exit')=='0':
            retrieve(pod,run_id)
            if run_id=='209':pod['calibration_complete']=True
            else:
                required=['209','210']
                if not receipt_path(pod,'209').exists():retrieve(pod,'209')
                if receipt_path(pod,'211').exists():required.append('211')
                delete(pod,required)
        elif s.get('worker_exit') is not None or s.get('setup_exit') not in [None,'0']:
            retrieve(pod,run_id,failed=True);delete(pod,['209'] if run_id=='209' else ['209','210'])
            pod['failed']=True
        pod['latest_status']=s
        pod.pop('transport_error',None)
    except Exception as error:
        pod['transport_error']=repr(error)
        pod.setdefault('error_since_epoch',time.time())
    return pod


def retire_superseded(pod):
    """Preserve partial evidence; the same complete arms are canonical in Run211."""
    if pod.get('deleted_epoch'):
        pod['calibration_complete']=True;return
    if pod.get('superseded_by_run'):return
    ctrl=control('209');rel=CAL.relative_to(ROOT).as_posix()
    ssh(pod,f'touch {REMOTE}/{rel}/stop-request\n')
    launch_record=HERE/f'launch-209-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
    if not launch_record.exists():
        setup_record=HERE/f'setup-209-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
        setup_pid=int(json.loads(setup_record.read_text())['pid'])
        script=f'''python3 - <<'PY'
from pathlib import Path
import os,signal,json
pid={setup_pid};root=Path('/proc')/str(pid);parents={{}}
if root.exists():
 assert b'{ctrl}/setup.sh' in (root/'cmdline').read_bytes()
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:parents[int(p.name)]=int((p/'stat').read_text().split(') ',1)[1].split()[1])
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 desc={{pid}}
 while True:
  more={{p for p,parent in parents.items() if parent in desc}}-desc
  if not more:break
  desc.update(more)
 for p in [pid]+sorted(desc-{{pid}},reverse=True):
  try:os.kill(p,signal.SIGKILL)
  except ProcessLookupError:pass
c=Path('{ctrl}')
(c/'setup.exit').write_text('125')
(c/'worker.exit').write_text('125')
(c/'superseded.json').write_text(json.dumps(dict(reason='Same complete calibration arms assigned to warm parallel Run211',scientific_updates_started=False)))
PY
'''
        ssh(pod,script)
    pod['superseded_by_run']='211'
    save(HERE/f'superseded-worker{pod["worker"]}-attempt{pod["attempt"]}.json',dict(pod_id=pod['id'],epoch=time.time(),replacement_run='211',stop_request=True))


def guard():
    requested=False
    while not (HERE/'closeout.json').exists():
        l=lease();now=time.time()
        cost=sum((p.get('deleted_epoch',now)-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in l['pods'])
        if cost>=l['limits']['graceful_compute_stop_usd'] or now>=l['deadline_epoch']-900:
            if not requested:
                def request(p):
                    if p.get('host') and not p.get('deleted_epoch'):
                        for run in [CAL,PROD]:
                            try:ssh(p,f'mkdir -p {REMOTE}/{run.relative_to(ROOT).as_posix()}\ntouch {REMOTE}/{run.relative_to(ROOT).as_posix()}/stop-request\n',timeout=20)
                            except Exception:pass
                with ThreadPoolExecutor(max_workers=19) as pool:list(pool.map(request,l['pods']))
                save(HERE/'budget_stop_request.json',dict(epoch=now,compute_estimate_usd=cost));requested=True
        if cost>=l['limits']['hard_compute_stop_usd'] or now>=l['deadline_epoch']:
            for p in l['pods']:
                if p.get('deleted_epoch'):continue
                state=cli('pod','get',p['id'],check=False)
                if state and state['name']==p['name'] and state['name'].startswith(PREFIX):cli('pod','stop',p['id'],check=False)
            save(HERE/'deadline_stop.json',dict(epoch=time.time(),compute_estimate_usd=cost,volume_retained_for_retrieval=True));return
        time.sleep(15)


def main():
    assert not (HERE/'closeout.json').exists(),'Already closed'
    if not (SCRATCH/'lease.json').exists():
        cfg=json.loads((CAL/'config.json').read_text())
        save(SCRATCH/'lease.json',dict(created_epoch=time.time(),deadline_epoch=time.time()+cfg['resource_limits']['max_lease_seconds'],
            limits=cfg['resource_limits'],pods=[]))
        with (SCRATCH/'guard.log').open('wb') as f:
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'guard'],stdout=f,stderr=subprocess.STDOUT,creationflags=NO_WINDOW)
        save(SCRATCH/'guard-process.json',dict(pid=p.pid))
    pool=ThreadPoolExecutor(max_workers=19)
    pending={};next_poll={}
    while True:
        l=lease()
        if not (PROD/'config.json').exists():
            for worker in range(19):
                if (HERE/'parallel_calibration_plan.json').exists() and worker in [1,2]:continue
                previous=[p for p in l['pods'] if p['worker']==worker]
                if previous and (not previous[-1].get('deleted_epoch') or previous[-1].get('calibration_complete')):continue
                assert len(previous)<3,('Retry bound',worker)
                allocate(worker,force_gpu='RTX 4090' if previous else None);l=lease()
        if (HERE/'parallel_calibration_ready.json').exists() and not (PROD/'config.json').exists():
            for pod in l['pods']:
                if pod['worker'] in [1,2] and pod['id'] not in pending:retire_superseded(pod)
        active=[p for p in l['pods'] if not p.get('deleted_epoch') and not (p['phase']=='209' and p.get('calibration_complete'))]
        for pod in active:
            if pod['id'] not in pending and time.time()>=next_poll.get(pod['id'],0):
                # Independent transfers must not hold back ready workers at a batch barrier.
                pending[pod['id']]=pool.submit(tick,dict(pod))
        changed=[]
        for pod_id,future in list(pending.items()):
            if future.done():
                changed.append(future.result());del pending[pod_id];next_poll[pod_id]=time.time()+30
        for pod in changed:
            if pod.get('superseded_by_run') and (pod.get('calibration_complete') or pod.get('deleted_epoch')):
                if not pod.get('deleted_epoch'):delete(pod,['209'])
                pod['calibration_complete']=True
        byid={p['id']:p for p in changed}
        for previous in l['pods']:
            if previous['id'] in byid and previous['phase']=='210':byid[previous['id']]['phase']='210'
        l['pods']=[byid.get(p['id'],p) for p in l['pods']]
        save(SCRATCH/'lease.json',l)
        progress=[dict(worker=p['worker'],gpu=p['gpu'],phase=p['phase'],deleted=bool(p.get('deleted_epoch')),
            progress=p.get('latest_status',{}).get('progress'),manifest=p.get('latest_status',{}).get('manifest'),
            error=p.get('transport_error')) for p in l['pods']]
        save(SCRATCH/'latest-status.json',dict(epoch=time.time(),workers=progress))
        parallel_ready=not (HERE/'parallel_calibration_plan.json').exists() or (HERE/'parallel_calibration_ready.json').exists()
        retired=not (HERE/'parallel_calibration_plan.json').exists() or all(p.get('deleted_epoch') for p in l['pods'] if p['worker'] in [1,2])
        def calibration_ready(worker):
            return (HERE/f'calibration-scalar-worker{worker}.json').exists() or any(p['worker']==worker and p.get('calibration_complete') for p in l['pods'])
        if not (PROD/'config.json').exists() and parallel_ready and retired and all(calibration_ready(w) for w in range(19)):
            production()
            for p in l['pods']:
                if not p.get('deleted_epoch'):p['phase']='210'
            save(SCRATCH/'lease.json',l)
        if (PROD/'config.json').exists() and all(p.get('deleted_epoch') for p in l['pods']):
            expected_workers=sorted({j['worker'] for j in json.loads((PROD/'config.json').read_text())['jobs']})
            for worker in expected_workers:
                successful=[p for p in l['pods'] if p['worker']==worker and receipt_path(p,'210').exists()
                    and json.loads(receipt_path(p,'210').read_text())['scientific_status']=='complete']
                assert successful,('Production failure needs an explicit infrastructure retry',worker)
            assert not any(p['name'].startswith(PREFIX) for p in cli('pod','list'))
            hours=sum((p['deleted_epoch']-p['created_epoch'])/3600 for p in l['pods'])
            cost=sum((p['deleted_epoch']-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in l['pods'])
            save(HERE/'closeout.json',dict(status='complete',pods=l['pods'],aggregate_gpu_hours=hours,
                compute_estimate_usd=cost,billing='provisional',cloud_storage_remaining_gb=0,network_volumes_created=0,
                maximum_total_usd=l['limits']['max_total_usd'],closed_epoch=time.time()))
            return
        time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','guard']);args=p.parse_args()
    guard() if args.action=='guard' else main()
