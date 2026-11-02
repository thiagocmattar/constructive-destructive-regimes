"""Single bounded parallel cohort; verified retrieval before task-Pod deletion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/212-2026-11-02-eos-backward-anchors'
SCRATCH=HERE/'scratch';SCRATCH.mkdir(exist_ok=True)
PREFIX='cdr-analysis035-'
spec=importlib.util.spec_from_file_location('transport034',ROOT/'analyses/034-2026-11-01-eos-adam-continuations/01_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.HERE=HERE;c.CAL=RUN;c.PROD=RUN;c.SCRATCH=SCRATCH;c.PREFIX=PREFIX
c.runpath=lambda _:RUN
c.control=lambda _:'/workspace/analysis035-control-212'
c.output=lambda pod,_: (f'/tmp/analysis035-artifacts/attempt-{pod["attempt"]:02d}/worker{pod["worker"]}'
    if pod.get('container_output') else f'{c.REMOTE}/{RUN.relative_to(ROOT).as_posix()}/artifacts/attempt-{pod["attempt"]:02d}/worker{pod["worker"]}')
save=c.save;sha=c.sha
CTRL=c.control('212')
RUNTIME='/workspace/analysis035-runtime'
LOCK=threading.RLock()


def lease():return json.loads((SCRATCH/'lease.json').read_text())


def preflight():
    gpus=c.cli('gpu','list','--include-unavailable')
    templates=c.cli('template','list','--type','official')
    image=next(x['imageName'] for x in templates if x['id']=='runpod-torch-v280')
    save(HERE/'preflight.json',dict(epoch=time.time(),gpus=gpus,image=image,
        baseline_pods=[dict(id=x['id'],name=x['name']) for x in c.cli('pod','list')],
        baseline_volumes=[{k:x[k] for k in ['id','name','size','dataCenterId']} for x in c.cli('network-volume','list')],
        cli_version=subprocess.check_output(['runpodctl','version'],text=True).strip()))


def allocate(worker):
    l=lease();attempt=1+sum(p['worker']==worker for p in l['pods'])
    assert attempt<=2
    catalog=c.cli('gpu','list','--include-unavailable')
    preferences=['RTX 4090','A100 SXM','A100 PCIe'] if worker<6 else ['RTX 5090','RTX 4090','A100 SXM','A100 PCIe']
    if attempt==2:preferences=['RTX 4090']
    image=json.loads((HERE/'preflight.json').read_text())['image']
    name=f'{PREFIX}worker{worker}-attempt{attempt}'
    assert not any(x['name']==name for x in c.cli('pod','list'))
    for choice in preferences:
        gpu=next(x for x in catalog if x['displayName']==choice)
        if not gpu['available'] and choice!='RTX 4090':continue
        assert gpu['securePricePerHr']<=1.79
        try:
            r=c.cli('pod','create','--name',name,'--image',image,'--gpu-id',gpu['gpuId'],
                '--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
                '--container-disk-in-gb','60','--volume-in-gb','35',
                '--ports','22/tcp,8888/http' if worker==12 else '22/tcp','--ssh')
            break
        except RuntimeError as error:
            if not any(x in str(error).lower() for x in ['no available','insufficient','capacity','no instances','instances available','not have the resources']):raise
            save(SCRATCH/f'capacity-{worker}-{choice.replace(" ","")}.json',dict(error=str(error)))
    else:return None
    p=dict(id=r['id'],name=name,worker=worker,attempt=attempt,created_epoch=time.time(),
           gpu=choice,price_per_gpu_hour=gpu['securePricePerHr'],phase='212',
           container_output=worker==3 and attempt>1)
    save(HERE/f'create-worker{worker}-attempt{attempt}.json',r)
    l=lease();l['pods'].append(p);save(SCRATCH/'lease.json',l)
    print(json.dumps(dict(created=p)),flush=True)
    return p


def setup(pod):
    marker=HERE/f'setup-212-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
    if marker.exists():return
    packet=next(x for x in json.loads((HERE/'packets-212.json').read_text())['packets'] if x['worker']==pod['worker'])
    if pod.get('container_output'):
        c.ssh(pod,f'mkdir -p /tmp/analysis035-control-212\nif [ ! -e {CTRL} ]; then ln -s /tmp/analysis035-control-212 {CTRL}; fi\n')
    c.ssh(pod,f'mkdir -p {CTRL} {RUNTIME}\n')
    present=c.ssh(pod,'sha256sum /tmp/analysis035-inputs.tar.gz 2>/dev/null || true\n').split()
    if not present or present[0]!=packet['sha256']:
        c.copy(pod,ROOT/packet['path'],'/tmp/analysis035-inputs.tar.gz',upload=True)
    c.copy(pod,HERE/'runtime_constraints.txt',CTRL+'/runtime_constraints.txt',upload=True)
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > {CTRL}/setup.exit' EXIT
printf '%s  %s\\n' '{packet['sha256']}' /tmp/analysis035-inputs.tar.gz | sha256sum -c -
mkdir -p {c.REMOTE}
tar -xzf /tmp/analysis035-inputs.tar.gz -C {c.REMOTE}
uv venv --system-site-packages /opt/analysis034-venv
uv pip install --python /opt/analysis034-venv/bin/python --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match --constraint {CTRL}/runtime_constraints.txt torch==2.11.0+cu130 numpy==2.5.0 scipy==1.18.1 cuda-bindings==13.4.4 cuda-pathfinder==1.8.4
/opt/analysis034-venv/bin/python -c 'import torch,numpy,scipy; assert torch.__version__=="2.11.0+cu130"; assert numpy.__version__=="2.5.0"; assert scipy.__version__=="1.18.1"; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
'''
    path=SCRATCH/f'setup-{pod["worker"]}-{pod["attempt"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
    c.copy(pod,path,CTRL+'/setup.sh',upload=True)
    pid=c.ssh(pod,f'setsid bash {CTRL}/setup.sh > {CTRL}/setup.log 2>&1 < /dev/null &\necho $!\n').strip()
    save(marker,dict(pid=pid,epoch=time.time(),packet_sha256=packet['sha256'],
        runtime_constraints_sha256=sha(HERE/'runtime_constraints.txt'),runtime_method='direct official index downloads; pinned source package versions'))


def launch(pod):
    marker=HERE/f'launch-212-worker{pod["worker"]}-attempt{pod["attempt"]}.json'
    if marker.exists():return
    out=c.output(pod,'212');rel=RUN.relative_to(ROOT).as_posix()
    script=f'''#!/bin/bash
set -uo pipefail
cd {c.REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM 6000 /opt/analysis034-venv/bin/python {rel}/01_worker.py --worker {pod['worker']} --output {out} > {CTRL}/worker.log 2>&1
echo $? > {CTRL}/worker.exit
'''
    path=SCRATCH/f'launch-{pod["worker"]}-{pod["attempt"]}.sh';path.write_text(script,encoding='utf-8',newline='\n')
    c.copy(pod,path,CTRL+'/worker.sh',upload=True)
    pid=c.ssh(pod,f'test ! -e {out}\nsetsid bash {CTRL}/worker.sh > {CTRL}/driver.log 2>&1 < /dev/null &\necho $!\n').strip()
    save(marker,dict(pid=pid,epoch=time.time(),config_sha256=sha(RUN/'config.json')))


def tick(pod):
    try:
        if not pod.get('host') and not c.connect(pod):return pod
        if pod.get('retiring'):
            ready=c.ssh(pod,f'if [ -f {CTRL}/retirement-capture.complete ]; then echo ready; fi\n').strip()
            if ready!='ready':return pod
        setup(pod)
        s=c.status(pod,'212');pod['latest_status']=s
        if s.get('setup_exit')=='0':launch(pod)
        terminal=s.get('worker_exit') is not None or s.get('setup_exit') not in [None,'0']
        if terminal:
            failed=s.get('worker_exit')!='0'
            c.retrieve(pod,'212',failed=failed)
            c.delete(pod,['212']);pod['failed']=failed
        pod.pop('transport_error',None)
    except Exception as error:
        pod['transport_error']=repr(error)
    return pod


def guard():
    while not (HERE/'closeout.json').exists():
        l=lease();now=time.time()
        cost=sum((p.get('deleted_epoch',now)-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in l['pods'])
        if cost>=12.5 or now>=l['deadline_epoch']-900:
            for p in l['pods']:
                if p.get('host') and not p.get('deleted_epoch'):
                    try:c.ssh(p,f'touch {c.REMOTE}/{RUN.relative_to(ROOT).as_posix()}/stop-request\n',timeout=20)
                    except Exception:pass
            save(HERE/'budget_stop_request.json',dict(epoch=now,compute_estimate_usd=cost))
        if cost>=14 or now>=l['deadline_epoch']:
            for p in l['pods']:
                if not p.get('deleted_epoch'):
                    actual=c.cli('pod','get',p['id'],check=False)
                    if actual and actual['name']==p['name'] and actual['name'].startswith(PREFIX):c.cli('pod','stop',p['id'],check=False)
            save(HERE/'deadline_stop.json',dict(epoch=now,compute_estimate_usd=cost,storage_preserved_for_retrieval=True));return
        time.sleep(10)


def main():
    assert not (HERE/'closeout.json').exists()
    if not (SCRATCH/'lease.json').exists():
        preflight()
        save(SCRATCH/'lease.json',dict(created_epoch=time.time(),deadline_epoch=time.time()+7200,
            runtime_name=secrets.token_hex(24)+'.tar.gz',pods=[]))
        with (SCRATCH/'guard.log').open('wb') as stream:
            p=subprocess.Popen([sys.executable,str(Path(__file__)),'guard'],stdout=stream,stderr=subprocess.STDOUT,creationflags=c.NO_WINDOW)
        save(SCRATCH/'guard-process.json',dict(pid=p.pid))
    pending={};next_poll={};next_allocate={}
    with ThreadPoolExecutor(max_workers=13) as pool:
        while True:
            l=lease()
            tf_ready=any(p['worker']==0 and p.get('latest_status',{}).get('progress',{}).get('phase') in ['gradient','backward_products'] for p in l['pods'])
            desired=[12,0,6]+(list(range(1,6))+list(range(7,12)) if tf_ready else list(range(7,12)))
            for worker in desired:
                prior=[p for p in l['pods'] if p['worker']==worker]
                if prior and not ((prior[-1].get('deleted_epoch') and prior[-1].get('failed') or prior[-1].get('retiring')) and len(prior)<2):continue
                if time.time()<next_allocate.get(worker,0):continue
                allocate(worker);l=lease();next_allocate[worker]=time.time()+30
            for pod in l['pods']:
                if not pod.get('deleted_epoch') and pod['id'] not in pending and time.time()>=next_poll.get(pod['id'],0):
                    pending[pod['id']]=pool.submit(tick,dict(pod))
            changed={}
            for pid,future in list(pending.items()):
                if future.done():
                    changed[pid]=future.result();del pending[pid];next_poll[pid]=time.time()+20
            # Merge into the latest lease so the seed server's published URL survives.
            with LOCK:
                l=lease();l['pods']=[changed.get(p['id'],p) for p in l['pods']];save(SCRATCH/'lease.json',l)
            save(SCRATCH/'latest-status.json',dict(epoch=time.time(),workers=l['pods']))
            if len({p['worker'] for p in l['pods']})==13 and all(p.get('deleted_epoch') for p in l['pods']):
                assert not any(x['name'].startswith(PREFIX) for x in c.cli('pod','list'))
                failures=[w for w in range(13) if not any(p['worker']==w and not p.get('failed') for p in l['pods'])]
                if failures and any(sum(p['worker']==w for p in l['pods'])<2 for w in failures):continue
                cost=sum((p['deleted_epoch']-p['created_epoch'])/3600*p['price_per_gpu_hour'] for p in l['pods'])
                save(HERE/'closeout.json',dict(status='failed' if failures else 'complete',failed_workers=failures,
                    pods=l['pods'],compute_estimate_usd=cost,billing='provisional',closed_epoch=time.time(),
                    cloud_storage_remaining_gb=0,network_volumes_created=0))
                return
            time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','guard']);a=p.parse_args()
    guard() if a.action=='guard' else main()
