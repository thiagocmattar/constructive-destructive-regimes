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
RUN=ROOT/'runs/205-2026-11-01-eos-exact-lag10'
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


def transport(pod,scp=False):
    return ['scp' if scp else 'ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=accept-new',
        '-o','UserKnownHostsFile='+str(ROOT/'cache/runpod/analysis032-known-hosts'),
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


def allocate():
    assert not (SCRATCH/'lease.json').exists(),'Do not duplicate task allocation'
    cfg=json.loads((RUN/'config.json').read_text());limits=cfg['resource_limits']
    gpus=cli('gpu','list');gpu=next(g for g in gpus if g['displayName']=='RTX 4090')
    templates=cli('template','list','--type','official')
    template=next(t for t in templates if t['id']=='runpod-torch-v280')
    assert gpu['available'] and gpu['securePricePerHr']<=.89
    save(HERE/'live_resources.json',dict(utc=time.time(),gpu=gpu,template=template,existing_pods=cli('pod','list')))
    lease=dict(created_epoch=time.time(),deadline_epoch=time.time()+limits['max_lease_seconds'],
        max_total_usd=limits['max_total_usd'],price_per_gpu_hour=gpu['securePricePerHr'],pods=[])
    save(SCRATCH/'lease.json',lease)
    with (SCRATCH/'guard.log').open('wb') as out:
        p=subprocess.Popen([sys.executable,str(Path(__file__)),'guard'],stdout=out,stderr=subprocess.STDOUT,creationflags=NO_WINDOW)
    save(SCRATCH/'guard-process.json',dict(pid=p.pid))
    for worker in range(3):
        name=f'cdr-analysis032-worker{worker}'
        result=cli('pod','create','--name',name,'--image',template['imageName'],'--gpu-id',gpu['gpuId'],
            '--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
            '--container-disk-in-gb',str(limits['container_gb']),'--volume-in-gb',str(limits['pod_volume_gb']),
            '--ports','22/tcp','--ssh')
        save(SCRATCH/f'create-worker{worker}.json',result)
        pod=dict(id=result['id'],name=name,worker=worker,created_epoch=time.time())
        lease['pods'].append(pod);save(SCRATCH/'lease.json',lease)
        print(json.dumps(pod),flush=True)


def connect():
    lease=loadlease()
    for pod in lease['pods']:
        info=cli('ssh','info',pod['id'],check=False)
        if not info or not info.get('ip'):
            state=cli('pod','get',pod['id'])
            info=state.get('ssh')
        if info and info.get('ip') and info.get('port'):
            pod.update(host=info['ip'],port=info['port'])
            save(SCRATCH/f'ssh-worker{pod["worker"]}.json',info)
    save(SCRATCH/'lease.json',lease)
    print(json.dumps([{k:v for k,v in p.items() if k in ['worker','id','host','port']} for p in lease['pods']]))


def stage_one(pod):
    worker=pod['worker'];packet=json.loads((HERE/'packet.json').read_text())
    ssh(pod,'mkdir -p /workspace/analysis032-control\n')
    copy(pod,ROOT/packet['path'],'/workspace/analysis032-inputs.tar.gz',upload=True)
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > /workspace/analysis032-control/setup.exit' EXIT
cd /workspace
printf '%s  %s\\n' '{packet['sha256']}' 'analysis032-inputs.tar.gz' | sha256sum -c -
mkdir -p {REMOTE}
tar -xzf analysis032-inputs.tar.gz -C {REMOTE}
python3 -m venv --system-site-packages /opt/analysis032-venv
/opt/analysis032-venv/bin/python -m pip install --no-cache-dir --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
/opt/analysis032-venv/bin/python -m pip install --no-cache-dir numpy==2.5.0 scipy==1.18.1
/opt/analysis032-venv/bin/python -c 'import torch,numpy,scipy; print(torch.__version__,torch.version.cuda,numpy.__version__,scipy.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
'''
    scriptfile=SCRATCH/f'setup-worker{worker}.sh';scriptfile.write_text(script,encoding='utf-8',newline='\n')
    copy(pod,scriptfile,'/workspace/analysis032-control/setup.sh',upload=True)
    output=ssh(pod,'nohup bash /workspace/analysis032-control/setup.sh > /workspace/analysis032-control/setup.log 2>&1 < /dev/null &\necho $!\n')
    save(HERE/f'staging-worker{worker}.json',dict(status='setup_detached',pid=output.strip(),packet_sha256=packet['sha256'],epoch=time.time()))
    print(json.dumps(dict(worker=worker,status='setup_detached')),flush=True)


def status_one(pod):
    if pod.get('deleted_epoch'):return dict(worker=pod['worker'],status='deleted')
    script=f'''python3 - <<'PY'
from pathlib import Path
import json,time,shutil
control=Path('/workspace/analysis032-control');out=Path('{REMOTE}/{REL}/artifacts/attempt-01/worker{pod['worker']}')
r={{'worker':{pod['worker']},'observed_epoch':time.time(),'disk_free_gb':shutil.disk_usage('/workspace').free/1e9}}
for name in ['setup','worker']:
 p=control/(name+'.exit')
 if p.exists():r[name+'_exit']=p.read_text().strip()
for name in ['progress','manifest']:
 p=out/(name+'.json')
 if p.exists():r[name]=json.loads(p.read_text())
if (control/'setup.log').exists():r['setup_tail']=(control/'setup.log').read_text(errors='replace')[-600:]
print(json.dumps(r))
PY
'''
    result=json.loads(ssh(pod,script))
    save(SCRATCH/f'status-worker{pod["worker"]}.json',result)
    return result


def launch():
    for pod in loadlease()['pods']:
        if (HERE/f'launch-worker{pod["worker"]}.json').exists():continue
        status=status_one(pod)
        if 'setup_exit' not in status:
            print(json.dumps(dict(worker=pod['worker'],status='setup_pending')),flush=True)
            continue
        assert status['setup_exit']=='0',status
        output=f'{REMOTE}/{REL}/artifacts/attempt-01/worker{pod["worker"]}'
        script=f'''#!/bin/bash
set -uo pipefail
cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM 3600 /opt/analysis032-venv/bin/python {REL}/01_worker.py --worker {pod['worker']} --output {output} > /workspace/analysis032-control/worker.log 2>&1
echo $? > /workspace/analysis032-control/worker.exit
'''
        path=SCRATCH/f'launch-worker{pod["worker"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
        copy(pod,path,'/workspace/analysis032-control/worker.sh',upload=True)
        result=ssh(pod,f'test ! -e {shlex.quote(output)}\nsetsid bash /workspace/analysis032-control/worker.sh > /workspace/analysis032-control/driver.log 2>&1 < /dev/null &\necho $!\n')
        save(HERE/f'launch-worker{pod["worker"]}.json',dict(pid=result.strip(),epoch=time.time(),output=output,config_sha256=sha(RUN/'config.json')))
        print(json.dumps(dict(worker=pod['worker'],pid=result.strip())),flush=True)


def retrieve_one(pod):
    worker=pod['worker'];out=f'{REMOTE}/{REL}/artifacts/attempt-01/worker{worker}'
    status=status_one(pod)
    assert status.get('worker_exit') is not None,'Worker still running'
    archive=f'/workspace/analysis032-worker{worker}.tar.gz'
    script=f'''set -e
cp /workspace/analysis032-control/setup.log {out}/setup.log
cp /workspace/analysis032-control/worker.log {out}/worker.log
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
    localarchive=ROOT/f'cache/analysis032-worker{worker}.tar.gz'
    copy(pod,archive,localarchive)
    assert sha(localarchive)==digest,'Archive transfer hash mismatch'
    destination=RUN/f'artifacts/attempt-01/worker{worker}';destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(localarchive) as tar:tar.extractall(destination,filter='data')
    inv=json.loads((destination/'inventory.json').read_text());total=0
    for item in inv['files']:
        p=(destination/item['path']).resolve()
        assert p.is_relative_to(destination.resolve())
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
        total+=item['bytes']
    manifest=json.loads((destination/'manifest.json').read_text())
    assert manifest['status']=='complete',manifest
    receipt=dict(worker=worker,verified=True,files=len(inv['files']),bytes=total,archive_sha256=digest,
        inventory_sha256=sha(destination/'inventory.json'),epoch=time.time())
    save(HERE/f'retrieval-worker{worker}.json',receipt)
    print(json.dumps(receipt),flush=True)


def cleanup():
    lease=loadlease()
    for pod in lease['pods']:
        if pod.get('deleted_epoch'):continue
        receipt_path=HERE/f'retrieval-worker{pod["worker"]}.json'
        if not receipt_path.exists():continue
        receipt=json.loads(receipt_path.read_text())
        assert receipt['verified']
        actual=cli('pod','get',pod['id'])
        assert actual['name']==pod['name'] and actual['name'].startswith('cdr-analysis032-')
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
    save(HERE/'closeout.json',dict(status='complete',pods=lease['pods'],aggregate_gpu_hours=hours,
        compute_estimate_usd=hours*lease['price_per_gpu_hour'],billing='provisional; storage and setup included in resource lifetime; posted billing may lag',
        cloud_storage_remaining_gb=0,network_volumes_created=0,maximum_total_usd=lease['max_total_usd'],closed_epoch=time.time()))
    print('Task pods deleted; no task cloud storage remains')


def guard():
    while True:
        lease=loadlease()
        if (HERE/'closeout.json').exists():return
        if time.time()>=lease['deadline_epoch']:
            outcomes=[]
            for pod in lease['pods']:
                if pod.get('deleted_epoch'):continue
                state=cli('pod','get',pod['id'],check=False)
                if state and state['name']==pod['name'] and state['name'].startswith('cdr-analysis032-'):
                    result=cli('pod','stop',pod['id'],check=False);outcomes.append(dict(id=pod['id'],result=result))
            save(HERE/'deadline_stop.json',dict(epoch=time.time(),outcomes=outcomes,pod_volume_retained=True))
            return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['allocate','connect','stage','status','launch','retrieve','cleanup','guard']);args=p.parse_args()
    if args.action=='allocate':allocate()
    elif args.action=='connect':connect()
    elif args.action=='stage':
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(stage_one,loadlease()['pods']))
    elif args.action=='status':
        with ThreadPoolExecutor(max_workers=3) as pool:
            for r in pool.map(status_one,loadlease()['pods']):
                with (HERE/'monitoring.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                progress=r.get('progress',{})
                print(json.dumps(dict(worker=r['worker'],status=r.get('status',r.get('manifest',{}).get('status','setup')),
                    setup_exit=r.get('setup_exit'),worker_exit=r.get('worker_exit'),
                    progress={k:progress[k] for k in ['phase','parent','anchor','offset','loss','seconds','completed_updates','supervised_tokens','peak_allocated_bytes'] if k in progress})),flush=True)
    elif args.action=='launch':launch()
    elif args.action=='retrieve':
        ready=[]
        for pod in loadlease()['pods']:
            if (HERE/f'retrieval-worker{pod["worker"]}.json').exists():continue
            if status_one(pod).get('worker_exit') is not None:ready.append(pod)
            else:print(json.dumps(dict(worker=pod['worker'],status='still_running')),flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(retrieve_one,ready))
    elif args.action=='cleanup':cleanup()
    elif args.action=='guard':guard()
