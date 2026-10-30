"""Bounded allocation, verified staging, detached execution and local retrieval."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('h024_control',HERE/'04_control.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)

def cli(*args):return json.loads(control.cli(*args).stdout)
def ssh(pod,lease,script,timeout=60):
    r=subprocess.run(control.transport(pod,lease)+[f"{pod['user']}@{pod['host']}",'bash -s'],input=script.encode('utf-8'),capture_output=True,timeout=timeout,creationflags=control.NO_WINDOW)
    if r.returncode:raise RuntimeError((r.stderr+r.stdout).decode(errors='replace'))
    return r.stdout.decode('utf-8')
def upload(pod,lease,local,remote):
    r=subprocess.run(control.transport(pod,lease,scp=True)+['-q',str(local),f"{pod['user']}@{pod['host']}:{remote}"],capture_output=True,text=True,timeout=3600,creationflags=control.NO_WINDOW)
    if r.returncode:raise RuntimeError(r.stderr)

def main():
    lease_path=HERE/'scratch/lease.json'
    resume=lease_path.exists()
    if resume and '--resume-provision' not in sys.argv:raise RuntimeError('Existing lease requires explicit infrastructure recovery')
    prices=cli('gpu','list')
    gpu=next(v for v in prices if v['gpuId']=='NVIDIA L40S')
    assert gpu['securePricePerHr']<=1.09
    dcs=[v['dataCenterId'] for v in gpu.get('dataCenterAvailability',[]) if v['stockStatus'].lower()!='none']
    if not dcs and not resume:raise RuntimeError('No live L40S placement; no allocation performed')
    dc=json.loads(lease_path.read_text())['volume']['dataCenterId'] if resume else ('US-MO-1' if 'US-MO-1' in dcs else dcs[0])
    control.save(HERE/'live_price.json',dict(observed_at_epoch=time.time(),gpu=gpu,datacenter=dc,storage_source='https://docs.runpod.io/pods/pricing',network_gb_month_usd=.07,container_gb_month_usd=.10,cli_version='3.0.1-551d6a4'))
    control.save(HERE/'scratch/authorization.json',dict(approved=True,source='User: Go on. You are allowed to run experiments using runpod.',agent_execution_cap=True,maximum_total_usd=14.6,maximum_hours=12,maximum_gpus=1))
    if resume:
        lease=json.loads(lease_path.read_text())
        assert lease['owner']=='Analysis030-current-task' and len(lease['pods'])<=2
        volume=lease['volume']
        dc=volume['dataCenterId']
    else:
        lease=dict(owner='Analysis030-current-task',created_at_epoch=time.time(),pods=[],ssh_key=str(Path.home()/'.ssh/id_ed25519_runpod'),storage_hourly_price=(200*.10+30*.1)/730,maximum_hourly_price=1.09+(200*.10+30*.1)/730)
        first_allocation=json.loads((HERE/'scratch/lease-attempt1.json').read_text())['created_at_epoch']
        lease['hard_deadline_epoch']=first_allocation+12*3600
        lease['training_deadline_epoch']=lease['hard_deadline_epoch']-1.5*3600
        volume=dict(kind='pod-volume',dataCenterId=dc,size=200,name='task pod attached persistent storage')
        lease['volume']=volume;control.save(lease_path,lease)
        guardlog=(HERE/'scratch/guard.log').open('w')
        guard=subprocess.Popen([sys.executable,str(HERE/'04_control.py'),'guard'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=guardlog,stderr=subprocess.STDOUT,creationflags=control.NO_WINDOW)
        control.save(HERE/'scratch/guard-process.json',dict(pid=guard.pid))
    try:
        for i in range(1):
            name=f'cdr-analysis030-worker{i}'
            # Persist the returned ID before readiness or SSH checks.
            pod=next((v for v in lease['pods'] if v['worker']==i),None)
            if pod is None:
                r=cli('pod','create','--name',name,'--template-id','runpod-torch-v280','--gpu-id',gpu['gpuId'],'--cloud-type','SECURE','--gpu-count','1','--data-center-ids',dc,'--container-disk-in-gb','30','--volume-in-gb','200','--ports','22/tcp','--min-cuda-version','12.8')
                pod=dict(id=r['id'],name=name,worker=i,hourly_price=1.09)
                lease['pods'].append(pod);control.save(lease_path,lease)
            deadline=time.time()+600
            while time.time()<deadline:
                details=cli('pod','get',pod['id'])
                assert details['name']==name and details['costPerHr']<=1.09
                info=details.get('ssh',{}) or {'ip':pod.get('host'),'port':pod.get('port')}
                control.save(HERE/f'scratch/ssh-info-{i}.json',info)
                # runpodctl3.0 SSH connection map (verified against current CLI source).
                if info.get('ip') and info.get('port'):
                    pod.update(host=info['ip'],port=info['port'],user='root')
                    control.save(lease_path,lease)
                    try:
                        ssh(pod,lease,'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n')
                        break
                    except (RuntimeError,subprocess.TimeoutExpired):pass
                time.sleep(10)
            else:raise TimeoutError('Pod SSH readiness deadline')
        first=lease['pods'][0]
        ssh(first,lease,'mkdir -p /workspace/constructive/worker-control\n')
        source=json.loads((HERE/'source_packet.json').read_text());inputs=json.loads((HERE/'input_packet.json').read_text())
        upload(first,lease,ROOT/source['path'],'/workspace/analysis030-source.tar.gz')
        ssh(first,lease,f"echo '{source['sha256']}  /workspace/analysis030-source.tar.gz' | sha256sum -c && tar -xzf /workspace/analysis030-source.tar.gz -C /workspace/constructive\n")
        def setup(pod):
            i=pod['worker']
            script=f'cd /workspace/constructive\nbash analyses/{HERE.name}/setup.sh {i} > worker-control/setup-{i}.log 2>&1\n'
            ssh(pod,lease,script,timeout=1800)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(setup,pod) for pod in lease['pods']]
            upload(first,lease,ROOT/inputs['path'],'/workspace/analysis030-inputs.tar')
            ssh(first,lease,f"echo '{inputs['sha256']}  /workspace/analysis030-inputs.tar' | sha256sum -c && tar -xf /workspace/analysis030-inputs.tar -C /workspace/constructive\n",timeout=1800)
            for f in futures:f.result()
        for pod in lease['pods']:
            i=pod['worker']
            script=f'''set -euo pipefail
cd /workspace/constructive
test ! -e worker-control/worker-{i}.lock
setsid /opt/h024-env/bin/python -u analyses/{HERE.name}/02_worker.py --worker {i} --deadline {lease['training_deadline_epoch']} > worker-control/worker-{i}.log 2>&1 < /dev/null &
echo $!
'''
            pid=ssh(pod,lease,script).strip()
            control.save(HERE/f'scratch/worker-{i}-launch.json',dict(pid=pid,at_epoch=time.time()))
        control.save(HERE/'execution.json',dict(status='launched',pods=[dict(id=p['id'],worker=p['worker'],name=p['name']) for p in lease['pods']],volume=volume,started_at_epoch=lease['created_at_epoch'],maximum_total_usd=15,hard_deadline_epoch=lease['hard_deadline_epoch'],training_deadline_epoch=lease['training_deadline_epoch']))
        subprocess.run([sys.executable,str(HERE/'04_control.py'),'monitor'],check=True,cwd=ROOT,creationflags=control.NO_WINDOW)
    except Exception as exc:
        control.save(HERE/'scratch/launch-error.json',dict(error=repr(exc),at_epoch=time.time()))
        for pod in lease['pods']:
            try:control.assert_owned(pod);control.cli('pod','stop',pod['id'])
            except Exception:pass
        raise

if __name__=='__main__':main()
