"""Task-scoped Runpod allocation, detached work, hash-verified retrieval and cleanup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/206-2026-11-01-eos-long-lags'
REMOTE='/workspace/constructive'
REL=RUN.relative_to(ROOT).as_posix()
SCRATCH=HERE/'scratch'
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');temp.replace(p)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def cli(*args,check=True,timeout=60):
    p=subprocess.run(['runpodctl',*args],capture_output=True,text=True,timeout=timeout,creationflags=NO_WINDOW)
    if p.returncode:
        if check:raise RuntimeError(p.stderr[:2000])
        return None
    return json.loads(p.stdout)


def loadlease():return json.loads((SCRATCH/'lease.json').read_text())


def run_rel(pod):return pod.get('run_path',REL)


def transport(pod,scp=False):
    return ['scp' if scp else 'ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/analysis033-known-hosts'),
        '-i',str(Path.home()/'.ssh/id_ed25519_runpod'),'-P' if scp else '-p',str(pod['port'])]


def ssh(pod,script,timeout=60):
    p=subprocess.run(transport(pod)+['root@'+pod['host'],'bash -s'],input=script.encode(),capture_output=True,timeout=timeout,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:]+p.stdout.decode(errors='replace')[-2000:])
    return p.stdout.decode()


def copy(pod,source,destination,upload=False):
    remote='root@'+pod['host']+':'
    args=[str(source),remote+destination] if upload else [remote+source,str(destination)]
    p=subprocess.run(transport(pod,True)+['-q',*args],capture_output=True,timeout=2700,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])


def allocate(workers,gpu_name=None,datacenter_ids=None,run_path=None,packet_manifest=None):
    assert not (HERE/'closeout.json').exists(),'Analysis033 is closed; changed or repeated scientific work needs a new run ID'
    cfg=json.loads((RUN/'config.json').read_text());limits=cfg['resource_limits']
    gpus=cli('gpu','list','--include-unavailable')
    choices=[next(g for g in gpus if g['displayName']==name) for name in ['RTX 5090','RTX 4090']]
    if gpu_name:choices=[g for g in choices if g['displayName']==gpu_name]
    templates=cli('template','list','--type','official')
    template=next(t for t in templates if t['id']=='runpod-torch-v280')
    if not (SCRATCH/'lease.json').exists():
        approval=json.loads((HERE/'authorization.json').read_text())
        assert approval['approved_maximum_usd']>=limits['max_total_usd']
        save(HERE/'live_resources.json',dict(utc=time.time(),gpus=choices,template=template,
            existing_pods=[{k:p[k] for k in ['id','name','costPerHr'] if k in p} for p in cli('pod','list')]))
        lease=dict(created_epoch=time.time(),deadline_epoch=time.time()+limits['max_lease_seconds'],
            max_total_usd=limits['max_total_usd'],graceful_compute_stop_usd=limits['graceful_compute_stop_usd'],pods=[])
        save(SCRATCH/'lease.json',lease)
        with (SCRATCH/'guard.log').open('wb') as out:
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'guard'],stdout=out,stderr=subprocess.STDOUT,creationflags=NO_WINDOW)
        save(SCRATCH/'guard-process.json',dict(pid=p.pid))
    else:lease=loadlease()
    for worker in workers:
        if any(p['worker']==worker and not p.get('deleted_epoch') for p in lease['pods']):continue
        assert sum(not p.get('deleted_epoch') for p in lease['pods'])<12,'Maximum12 concurrent GPUs'
        attempt=1+sum(p['worker']==worker for p in lease['pods'])
        name=f'cdr-analysis033-worker{worker}'+(f'-attempt{attempt}' if attempt>1 else '')
        for gpu in choices:
            assert gpu['securePricePerHr']<={'RTX 5090':1.19,'RTX 4090':.89}[gpu['displayName']]
            if not gpu['available']:continue
            try:
                result=cli('pod','create','--name',name,'--image',template['imageName'],'--gpu-id',gpu['gpuId'],
                    '--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
                    '--container-disk-in-gb',str(limits['container_gb']),'--volume-in-gb',str(max(10,limits['pod_volume_gb'])),
                    '--ports','22/tcp','--ssh',*(['--data-center-ids',','.join(datacenter_ids)] if datacenter_ids else []))
                break
            except RuntimeError as error:
                # Never retry an ambiguous create/network failure: a resource might exist.
                if not any(s in str(error).lower() for s in ['no available','insufficient','capacity','no instances','not have the resources']):raise
                save(SCRATCH/f'capacity-worker{worker}-{gpu["displayName"].replace(" ","")}.json',dict(error=str(error)))
        else:raise RuntimeError('No capacity in authorized GPU classes')
        save(SCRATCH/f'create-worker{worker}.json',result)
        pod=dict(id=result['id'],name=name,worker=worker,created_epoch=time.time(),
                 gpu=gpu['displayName'],price_per_gpu_hour=gpu['securePricePerHr'],pod_volume_gb=max(10,limits['pod_volume_gb']),attempt=attempt)
        if run_path:pod.update(run_path=run_path,packet_manifest=packet_manifest,reused_runtime=True,warmup_required=True)
        lease['pods'].append(pod);save(SCRATCH/'lease.json',lease)
        print(json.dumps(pod),flush=True)


def connect():
    lease=loadlease()
    for pod in lease['pods']:
        if pod.get('deleted_epoch') or pod.get('host'):continue
        info=cli('ssh','info',pod['id'],check=False)
        if not info or not info.get('ip'):
            state=cli('pod','get',pod['id'])
            info=state.get('ssh')
        if info and info.get('ip') and info.get('port'):
            pod.update(host=info['ip'],port=info['port'])
            save(SCRATCH/f'ssh-worker{pod["worker"]}.json',info)
    save(SCRATCH/'lease.json',lease)
    print(json.dumps([{k:v for k,v in p.items() if k in ['worker','id','host','port']} for p in lease['pods']]))


def warm_one(pod):
    worker=pod['worker']
    if (HERE/f'warmup-worker{worker}.json').exists():return
    bundle=ROOT/'cache/analysis033-wheels.tar';assert bundle.exists()
    ssh(pod,'mkdir -p /workspace/analysis033-control\n')
    copy(pod,bundle,'/tmp/analysis033-wheels.tar',upload=True)
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > /workspace/analysis033-control/runtime.exit' EXIT
printf '%s  %s\\n' '{sha(bundle)}' /tmp/analysis033-wheels.tar | sha256sum -c -
mkdir -p /tmp/analysis033-wheels
tar -xf /tmp/analysis033-wheels.tar -C /tmp/analysis033-wheels
python3 -m venv --system-site-packages /opt/analysis033-venv
/opt/analysis033-venv/bin/python -m pip install --no-index --find-links /tmp/analysis033-wheels cuda-bindings==13.4.4 cuda-pathfinder==1.8.4 numpy==2.5.0 scipy==1.18.1
/opt/analysis033-venv/bin/python -m pip install --no-cache-dir --timeout 45 --retries 2 --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
/opt/analysis033-venv/bin/python -c 'import torch,numpy,scipy; assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name())'
'''
    path=SCRATCH/f'warmup-worker{worker}.sh';path.write_text(script,encoding='utf-8',newline='\n')
    copy(pod,path,'/workspace/analysis033-control/runtime.sh',upload=True)
    result=ssh(pod,'nohup bash /workspace/analysis033-control/runtime.sh > /workspace/analysis033-control/runtime.log 2>&1 < /dev/null &\necho $!\n')
    save(HERE/f'warmup-worker{worker}.json',dict(epoch=time.time(),pid=result.strip(),bundle_sha256=sha(bundle)))


def stage_one(pod):
    worker=pod['worker']
    if (HERE/f'staging-worker{worker}.json').exists():return
    packet_file=pod.get('packet_manifest','parallel_packets.json' if pod.get('run_path') else 'packets.json')
    if not (HERE/packet_file).exists():return
    if pod.get('warmup_required'):
        ready=ssh(pod,"test -f /workspace/analysis033-control/runtime.exit && cat /workspace/analysis033-control/runtime.exit || true\n",timeout=20).strip()
        if ready!='0':return
    packet=next(p for p in json.loads((HERE/packet_file).read_text())['packets'] if p['worker']==worker)
    ssh(pod,'mkdir -p /workspace/analysis033-control\n')
    copy(pod,ROOT/packet['path'],'/tmp/analysis033-inputs.tar.gz',upload=True)
    install='''python3 -m venv --system-site-packages /opt/analysis033-venv
/opt/analysis033-venv/bin/python -m pip install --no-cache-dir --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
/opt/analysis033-venv/bin/python -m pip install --no-cache-dir numpy==2.5.0 scipy==1.18.1
''' if not pod.get('reused_runtime') else ''
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > /workspace/analysis033-control/setup.exit' EXIT
cd /workspace
printf '%s  %s\\n' '{packet['sha256']}' '/tmp/analysis033-inputs.tar.gz' | sha256sum -c -
mkdir -p {REMOTE}
tar -xzf /tmp/analysis033-inputs.tar.gz -C {REMOTE}
{install}
/opt/analysis033-venv/bin/python -c 'import torch,numpy,scipy; print(torch.__version__,torch.version.cuda,numpy.__version__,scipy.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
'''
    scriptfile=SCRATCH/f'setup-worker{worker}.sh';scriptfile.write_text(script,encoding='utf-8',newline='\n')
    copy(pod,scriptfile,'/workspace/analysis033-control/setup.sh',upload=True)
    output=ssh(pod,'nohup bash /workspace/analysis033-control/setup.sh > /workspace/analysis033-control/setup.log 2>&1 < /dev/null &\necho $!\n')
    save(HERE/f'staging-worker{worker}.json',dict(status='setup_detached',pid=output.strip(),packet_sha256=packet['sha256'],epoch=time.time()))
    print(json.dumps(dict(worker=worker,status='setup_detached')),flush=True)


def status_one(pod):
    if pod.get('deleted_epoch'):return dict(worker=pod['worker'],status='deleted')
    if not pod.get('host'):return dict(worker=pod['worker'],status='no_ssh_yet')
    script=f'''python3 - <<'PY'
from pathlib import Path
import json,time,shutil
control=Path('/workspace/analysis033-control');out=Path('{REMOTE}/{run_rel(pod)}/artifacts/attempt-{pod.get('attempt',1):02d}/worker{pod['worker']}')
r={{'worker':{pod['worker']},'observed_epoch':time.time(),'disk_free_gb':shutil.disk_usage('/workspace').free/1e9}}
for name in ['setup','worker','calibration','runtime']:
 p=control/(name+'.exit')
 if p.exists():r[name+'_exit']=p.read_text().strip()
for name in ['progress','manifest']:
 p=out/(name+'.json')
 if p.exists():r[name]=json.loads(p.read_text())
if (control/'setup.log').exists():r['setup_tail']=(control/'setup.log').read_text(errors='replace')[-600:]
if (control/'worker.log').exists():r['worker_tail']=(control/'worker.log').read_text(errors='replace')[-700:]
print(json.dumps(r))
PY
'''
    result=json.loads(ssh(pod,script))
    save(SCRATCH/f'status-worker{pod["worker"]}.json',result)
    return result


def launch(workers):
    decision=json.loads((HERE/'calibration_decision.json').read_text())
    for pod in loadlease()['pods']:
        if pod['worker'] not in workers or pod.get('deleted_epoch'):continue
        if (HERE/f'launch-worker{pod["worker"]}.json').exists():continue
        status=status_one(pod)
        if 'setup_exit' not in status:
            print(json.dumps(dict(worker=pod['worker'],status='setup_pending')),flush=True)
            continue
        assert status['setup_exit']=='0',status
        rel=run_rel(pod)
        output=f'{REMOTE}/{rel}/artifacts/attempt-{pod.get("attempt",1):02d}/worker{pod["worker"]}'
        script=f'''#!/bin/bash
set -uo pipefail
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM 13600 /opt/analysis033-venv/bin/python {rel}/01_worker.py --worker {pod['worker']} --physical {decision['selected_physical']} --output {output} > /workspace/analysis033-control/worker.log 2>&1
echo $? > /workspace/analysis033-control/worker.exit
'''
        path=SCRATCH/f'launch-worker{pod["worker"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
        copy(pod,path,'/workspace/analysis033-control/worker.sh',upload=True)
        result=ssh(pod,f'test ! -e {shlex.quote(output)}\nsetsid bash /workspace/analysis033-control/worker.sh > /workspace/analysis033-control/driver.log 2>&1 < /dev/null &\necho $!\n')
        save(HERE/f'launch-worker{pod["worker"]}.json',dict(pid=result.strip(),epoch=time.time(),output=output,config_sha256=sha(ROOT/rel/'config.json')))
        print(json.dumps(dict(worker=pod['worker'],pid=result.strip())),flush=True)


def calibrate():
    pod=next(p for p in loadlease()['pods'] if p['worker']==0)
    status=status_one(pod);assert status.get('setup_exit')=='0',status
    if not (HERE/'calibration_launch.json').exists():
        out=f'{REMOTE}/{REL}/artifacts/calibration/worker0'
        script=f'''#!/bin/bash
set -uo pipefail
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/opt/analysis033-venv/bin/python {REL}/01_worker.py --worker 0 --calibrate --output {out} > /workspace/analysis033-control/calibration.log 2>&1
echo $? > /workspace/analysis033-control/calibration.exit
'''
        path=SCRATCH/'calibration.sh';path.write_text(script,encoding='utf-8',newline='\n')
        copy(pod,path,'/workspace/analysis033-control/calibration.sh',upload=True)
        result=ssh(pod,'setsid bash /workspace/analysis033-control/calibration.sh > /workspace/analysis033-control/calibration-driver.log 2>&1 < /dev/null &\necho $!\n')
        save(HERE/'calibration_launch.json',dict(epoch=time.time(),pid=result.strip()))
        print('Calibration detached');return
    status=status_one(pod)
    if 'calibration_exit' not in status:print('Calibration pending');return
    assert status['calibration_exit']=='0',ssh(pod,'tail -25 /workspace/analysis033-control/calibration.log')
    out=f'{REMOTE}/{REL}/artifacts/calibration/worker0'
    decision=json.loads(ssh(pod,f'cat {out}/calibration.json'))
    save(HERE/'calibration_decision.json',decision)
    print(json.dumps(decision))


def retrieve_one(pod,allow_failed=False):
    worker=pod['worker'];out=f'{REMOTE}/{run_rel(pod)}/artifacts/attempt-{pod.get("attempt",1):02d}/worker{worker}'
    status=status_one(pod)
    assert status.get('worker_exit') is not None or (allow_failed and status.get('setup_exit') not in [None,'0']),'Worker still running'
    if allow_failed:
        ssh(pod,f'''mkdir -p {out}
cp /workspace/analysis033-control/setup.log {out}/setup.log
if test -f /workspace/analysis033-control/worker.log; then cp /workspace/analysis033-control/worker.log {out}/worker.log; else touch /workspace/analysis033-control/worker.log; fi
if ! test -f {out}/manifest.json; then printf '%s\\n' '{{"status":"infrastructure_failed","worker":{worker}}}' > {out}/manifest.json; fi
''')
    archive=f'/tmp/analysis033-worker{worker}.tar.gz'
    script=f'''set -e
mkdir -p {out}/control
cp -a /workspace/analysis033-control/. {out}/control/
cp /workspace/analysis033-control/setup.log {out}/setup.log
cp /workspace/analysis033-control/worker.log {out}/worker.log
if test -d {REMOTE}/{REL}/artifacts/calibration/worker0 && test {worker} = 0; then cp -r {REMOTE}/{REL}/artifacts/calibration/worker0 {out}/calibration; cp /workspace/analysis033-control/calibration.log {out}/calibration/setup-driver.log; fi
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('{out}')
files=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='inventory.json':
  with f.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
  files.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h))
(p/'inventory.json').write_text(json.dumps(dict(files=files),indent=2)+'\\n')
PY
tar -I 'gzip -1' -cf {archive} -C {out} .
sha256sum {archive}
'''
    digest=ssh(pod,script,timeout=300).split()[0]
    localarchive=ROOT/f'cache/analysis033-worker{worker}.tar.gz'
    copy(pod,archive,localarchive)
    assert sha(localarchive)==digest,'Archive transfer hash mismatch'
    destination=ROOT/run_rel(pod)/f'artifacts/attempt-{pod.get("attempt",1):02d}/worker{worker}';destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(localarchive) as tar:tar.extractall(destination,filter='data')
    inv=json.loads((destination/'inventory.json').read_text());total=0
    for item in inv['files']:
        p=(destination/item['path']).resolve()
        assert p.is_relative_to(destination.resolve())
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
        total+=item['bytes']
    manifest=json.loads((destination/'manifest.json').read_text())
    assert manifest['status']=='complete' or (allow_failed and manifest['status'] in ['failed','infrastructure_failed']),manifest
    receipt=dict(worker=worker,pod_id=pod['id'],attempt=pod.get('attempt',1),
        artifact_root=destination.relative_to(ROOT).as_posix(),verified=True,files=len(inv['files']),bytes=total,archive_sha256=digest,
        inventory_sha256=sha(destination/'inventory.json'),epoch=time.time(),scientific_status=manifest['status'])
    save(HERE/f'retrieval-worker{worker}.json',receipt)
    print(json.dumps(receipt),flush=True)


def stop_completed_first_window(pod,record_name='serial_worker11_stop.json',reason=None,touch_stop=False):
    """Stop the serial worker after its first independent ResNet profile."""
    receipt=json.loads((HERE/f'launch-worker{pod["worker"]}.json').read_text())
    script=f'''python3 - <<'PY'
from pathlib import Path
import os,signal,json
output={json.dumps(receipt['output'])}
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if len(a)>1 and a[0].split(b'/')[-1].startswith(b'python') and a[1].endswith(b'/01_worker.py') and output.encode() in a:
  os.kill(int(p.name),signal.SIGTERM);matches.append(int(p.name))
print(json.dumps(matches))
PY
'''
    if touch_stop:ssh(pod,f'touch {REMOTE}/{run_rel(pod)}/stop-request\n',timeout=20)
    result=ssh(pod,script,timeout=20)
    save(HERE/record_name,dict(epoch=time.time(),pod_id=pod['id'],reason=reason or 'Later two independent windows are assigned to Run207; first completed window retained',signalled_pids=json.loads(result)))


def reuse(pod,new_worker,target_run_path=None,packet_manifest=None):
    lease=loadlease();actual=next(p for p in lease['pods'] if p['id']==pod['id'])
    old=actual['worker'];receipt=json.loads((HERE/f'retrieval-worker{old}.json').read_text())
    assert receipt['verified'] and receipt['pod_id']==actual['id'] and actual['gpu']=='RTX 4090'
    assert status_one(actual).get('worker_exit')=='0'
    ssh(actual,f'mv /workspace/analysis033-control /workspace/analysis033-control-role{old}\nmkdir -p /workspace/analysis033-control\n',timeout=20)
    actual.setdefault('role_history',[]).append(dict(worker=old,attempt=actual.get('attempt',1),
        artifact_root=receipt['artifact_root'],receipt=f'retrieval-worker{old}.json',finished_epoch=time.time()))
    plan=json.loads((HERE/'parallel_plan.json').read_text())
    actual.update(worker=new_worker,attempt=1,run_path=target_run_path or plan['run_path'],reused_runtime=True)
    actual.pop('warmup_required',None)
    if packet_manifest:actual['packet_manifest']=packet_manifest
    save(SCRATCH/'lease.json',lease)
    save(HERE/f'reuse-worker{old}-to{new_worker}.json',dict(epoch=time.time(),pod_id=actual['id'],old_worker=old,new_worker=new_worker,scientific_run=actual['run_path'].split('/')[1][:3],previous_receipt_sha256=sha(HERE/f'retrieval-worker{old}.json')))
    print(json.dumps(dict(reused_pod=actual['id'],old_worker=old,new_worker=new_worker)),flush=True)


def cleanup():
    lease=loadlease()
    for pod in lease['pods']:
        if pod.get('deleted_epoch'):continue
        receipt_path=HERE/f'retrieval-worker{pod["worker"]}.json'
        if not receipt_path.exists():continue
        receipt=json.loads(receipt_path.read_text())
        assert receipt['verified'] and receipt['pod_id']==pod['id'],'Receipt does not belong to this Pod'
        actual=cli('pod','get',pod['id'])
        assert actual['name']==pod['name'] and actual['name'].startswith('cdr-analysis033-')
        save(HERE/f'final-pod-worker{pod["worker"]}.json',{k:actual[k] for k in
            ['id','name','costPerHr','createdAt','gpuCount','gpuId','imageName','containerDiskInGb','volumeInGb','desiredStatus','runtimeStatus'] if k in actual})
        cli('pod','delete',pod['id'])
        probe=subprocess.run(['runpodctl','pod','get',pod['id']],capture_output=True,text=True,creationflags=NO_WINDOW,timeout=45)
        assert probe.returncode!=0 and json.loads(probe.stderr.splitlines()[0]).get('code')=='not_found','Deletion absence not confirmed'
        pod['deleted_epoch']=time.time();save(SCRATCH/'lease.json',lease)
    remaining=[p['worker'] for p in lease['pods'] if not p.get('deleted_epoch')]
    if remaining:
        print(json.dumps(dict(status='verified_workers_deleted',remaining_workers=remaining)),flush=True)
        return
    hours=sum((p['deleted_epoch']-p['created_epoch'])/3600 for p in lease['pods'])
    compute=sum((p['deleted_epoch']-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in lease['pods'])
    save(HERE/'closeout.json',dict(status='complete',pods=lease['pods'],aggregate_gpu_hours=hours,
        compute_estimate_usd=compute,billing='provisional; posted billing and storage may lag',
        cloud_storage_remaining_gb=0,network_volumes_created=0,maximum_total_usd=lease['max_total_usd'],closed_epoch=time.time()))
    print('Task pods deleted; no task cloud storage remains')


def guard():
    requested=False
    while True:
        lease=loadlease()
        if (HERE/'closeout.json').exists():return
        cost=sum((p.get('deleted_epoch',time.time())-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in lease['pods'])
        if (cost>=lease['graceful_compute_stop_usd'] or time.time()>=lease['deadline_epoch']-600) and not requested:
            def request(pod):
                if pod.get('host') and not pod.get('deleted_epoch'):
                    try:ssh(pod,f'touch {REMOTE}/{run_rel(pod)}/stop-request',timeout=20)
                    except Exception:pass
            with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(request,lease['pods']))
            save(HERE/'budget_stop_request.json',dict(epoch=time.time(),compute_estimate_usd=cost,
                reason='compute_budget' if cost>=lease['graceful_compute_stop_usd'] else 'deadline_retrieval_margin'))
            requested=True
        if time.time()>=lease['deadline_epoch'] or cost>=lease.get('hard_compute_stop_usd',lease['max_total_usd']-1):
            def stop_owned(pod):
                if pod.get('deleted_epoch'):return None
                state=cli('pod','get',pod['id'],check=False)
                if state and state['name']==pod['name'] and state['name'].startswith('cdr-analysis033-'):
                    result=cli('pod','stop',pod['id'],check=False);return dict(id=pod['id'],result=result)
            with ThreadPoolExecutor(max_workers=12) as pool:outcomes=list(pool.map(stop_owned,lease['pods']))
            save(HERE/'deadline_stop.json',dict(epoch=time.time(),outcomes=outcomes,pod_volume_retained=True))
            return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['allocate','connect','stage','status','calibrate','launch','retrieve','cleanup','guard'])
    p.add_argument('--workers',default='all');args=p.parse_args()
    workers=list(range(12)) if args.workers=='all' else [int(s) for s in args.workers.split(',')]
    if args.action=='allocate':allocate(workers)
    elif args.action=='connect':connect()
    elif args.action=='stage':
        with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(stage_one,[pod for pod in loadlease()['pods'] if pod['worker'] in workers and not pod.get('deleted_epoch')]))
    elif args.action=='status':
        with ThreadPoolExecutor(max_workers=12) as pool:
            for r in pool.map(status_one,loadlease()['pods']):
                with (HERE/'monitoring.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                progress=r.get('progress',{})
                print(json.dumps(dict(worker=r['worker'],status=r.get('status',r.get('manifest',{}).get('status','setup')),
                    setup_exit=r.get('setup_exit'),worker_exit=r.get('worker_exit'),
                    progress={k:progress[k] for k in ['phase','parent','anchor','lag','loss','seconds','completed_updates','supervised_tokens','peak_allocated_bytes','resolved_physical'] if k in progress})),flush=True)
    elif args.action=='launch':launch(workers)
    elif args.action=='calibrate':calibrate()
    elif args.action=='retrieve':
        ready=[]
        for pod in loadlease()['pods']:
            if pod.get('deleted_epoch'):continue
            if (HERE/f'retrieval-worker{pod["worker"]}.json').exists():continue
            if status_one(pod).get('worker_exit') is not None:ready.append(pod)
            else:print(json.dumps(dict(worker=pod['worker'],status='still_running')),flush=True)
        with ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(retrieve_one,ready))
    elif args.action=='cleanup':cleanup()
    elif args.action=='guard':guard()
