"""Run214-only Runpod lifecycle, detached guard, and verified retrieval."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'
SCRATCH=HERE/'scratch';SCRATCH.mkdir(exist_ok=True)
REMOTE='/workspace/constructive214'
CONTROL='/workspace/cdr214-control'
REL=RUN.relative_to(ROOT).as_posix()
PREFIX='cdr-analysis037-'
NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0)
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n');temporary.replace(path)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lease():return read(SCRATCH/'lease.json')


def cli(*args):
    p=subprocess.run(['runpodctl',*args],capture_output=True,text=True,timeout=90,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError((p.stderr+p.stdout)[-2500:])
    return json.loads(p.stdout)


def transport(pod,scp=False):
    return ['scp' if scp else 'ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes',
        '-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
        '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(SCRATCH/'known-hosts'),
        '-i',pod.get('ssh_key_path',str(Path.home()/'.ssh/id_ed25519_runpod')),'-P' if scp else '-p',str(pod['port'])]


def ssh(pod,script,timeout=60):
    p=subprocess.run(transport(pod)+['root@'+pod['host'],'bash -s'],input=script.encode(),
        capture_output=True,timeout=timeout,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr).decode(errors='replace')[-2500:])
    return p.stdout.decode()


def copy(pod,local,remote,upload=True):
    endpoint='root@'+pod['host']+':'+remote
    args=[str(local),endpoint] if upload else [endpoint,str(local)]
    p=subprocess.run(transport(pod,True)+['-q',*args],capture_output=True,text=True,timeout=900,creationflags=NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])


def owned(pod):
    actual=cli('pod','get',pod['id'])
    assert actual['name']==pod['name'] and actual['name'].startswith(PREFIX)
    return actual


def spend(state):
    return sum((p.get('deleted_epoch',time.time())-p['created_epoch'])/3600*(p['price_per_gpu_hour']+65*.10/720) for p in state['pods'])


def initialize():
    assert not (SCRATCH/'lease.json').exists()
    preflight=read(HERE/'runpod_preflight.json')
    for packet in preflight['packets']:assert file_hash(ROOT/packet['path'])==packet['sha256']
    authorization=dict(utc=stamp(),user_reply='approved',max_total_usd=40,max_wall_hours=4,
        max_concurrent_gpus=6,max_gpu_usd_per_hour=1.19,calibration_max_minutes=20,retrieval_margin_minutes=30,
        scientific_stop_usd=32,config_sha256=file_hash(RUN/'config.json'),input_manifest_sha256=file_hash(RUN/'input_manifest.json'))
    save(HERE/'authorization.json',authorization)
    state=dict(authorization=authorization,created_utc=stamp(),start_epoch=None,deadline_epoch=None,pods=[],
        baseline_pods=cli('pod','list'),baseline_volumes=cli('network-volume','list'))
    save(SCRATCH/'lease.json',state)
    with (SCRATCH/'guard.log').open('wb') as log:
        guard_process=subprocess.Popen([sys.executable,str(Path(__file__)),'guard'],stdout=log,stderr=subprocess.STDOUT,creationflags=NO_WINDOW)
    save(SCRATCH/'guard-process.json',dict(pid=guard_process.pid,utc=stamp()))


def allocate(task,seed,gpu='RTX 5090',cloud='SECURE',gpu_count=1,role=None):
    state=lease();assert sum(p.get('gpu_count',1) for p in state['pods'] if not p.get('deleted_epoch'))+gpu_count<=8
    active=[p for p in state['pods'] if p['task']==task and p['seed']==seed and not p.get('deleted_epoch')]
    if role in ('initial-continuation-helper','late-continuation-helper'):
        assert len(active)==1 and active[0].get('production_pid') and not active[0].get('role')
    else:assert not active
    catalog=cli('gpu','list','--include-unavailable');entry=next(x for x in catalog if x['displayName']==gpu)
    price=entry['securePricePerHr' if cloud=='SECURE' else 'communityPricePerHr']
    assert price and price<=4.0 and (state['deadline_epoch'] is None or time.time()<state['deadline_epoch']-1800)
    if price*gpu_count>1.19:
        remaining=(state['deadline_epoch']-time.time())/3600
        rate=sum(p['price_per_gpu_hour']+65*.10/720 for p in state['pods'] if not p.get('deleted_epoch'))
        assert spend(state)+remaining*(rate+price*gpu_count+65*.10/720)<38,'Worst-case task lease would exceed the approved total.'
    attempt=1+sum(p['task']==task and p['seed']==seed for p in state['pods'])
    name=f'{PREFIX}{task}-seed{seed}-a{attempt}'
    assert not any(p['name']==name for p in cli('pod','list'))
    template=next(t for t in cli('template','list','--type','official') if t['id']=='runpod-torch-v280')
    before=time.time()
    try:
        result=cli('pod','create','--name',name,'--image',template['imageName'],'--gpu-id',entry['gpuId'],
            '--gpu-count',str(gpu_count),'--cloud-type',cloud,'--min-cuda-version','12.8','--container-disk-in-gb','30',
            '--volume-in-gb','35','--ports','22/tcp','--ssh')
    except RuntimeError as error:
        if 'no longer any instances' not in str(error):raise
        save(HERE/f'capacity-attempt-{int(before)}.json',dict(utc=stamp(),task=task,seed=seed,gpu=gpu,cloud=cloud,price=price,error=str(error)))
        print(json.dumps(dict(status='capacity_unavailable',gpu=gpu,cloud=cloud)),flush=True)
        return
    pod=dict(id=result['id'],name=name,task=task,seed=seed,attempt=attempt,created_epoch=before,
        gpu=gpu,cloud=cloud,gpu_count=gpu_count,unit_price_per_gpu_hour=price,
        price_per_gpu_hour=price*gpu_count,image=template['imageName'])
    if role:pod.update(role=role,defer_stream=True)
    state=lease()
    if state['start_epoch'] is None:state.update(start_epoch=before,deadline_epoch=before+4*3600)
    state['pods'].append(pod);save(SCRATCH/'lease.json',state)
    save(HERE/f'create-{task}-{seed}-a{attempt}.json',dict(utc=stamp(),catalog_entry=entry,response=result))
    print(json.dumps(pod),flush=True)
    return pod


def connect(pod):
    if pod.get('host') and pod.get('port'):return
    state=owned(pod);info=state.get('ssh') or cli('ssh','info',pod['id'])
    if not info.get('ip') or not info.get('port'):raise RuntimeError('SSH route not ready.')
    pod.update(host=info['ip'],port=info['port'])
    key=info.get('ssh_key',{}).get('path')
    if key and Path(key).is_file():pod['ssh_key_path']=key
    ssh(pod,'echo ready\n',timeout=25)


def setup(pod):
    connect(pod)
    packets=read(HERE/'transfer_packets.json') if (HERE/'transfer_packets.json').exists() else read(HERE/'runpod_preflight.json')
    packet=next(x for x in packets['packets'] if x['task']==pod['task'])
    ssh(pod,f'mkdir -p {CONTROL}\n')
    copy(pod,ROOT/packet['path'],'/workspace/cdr214-inputs.tar')
    wheel_bundle=SCRATCH/'linux-wheels.tar'
    install='python3 -m pip --python /opt/cdr214-venv install --no-cache-dir --disable-pip-version-check numpy==2.5.0 scipy==1.18.1'
    wheel_setup=''
    if wheel_bundle.exists():
        copy(pod,wheel_bundle,'/workspace/cdr214-wheels.tar')
        wheel_setup=f"echo '{file_hash(wheel_bundle)}  /workspace/cdr214-wheels.tar' | sha256sum -c -\nmkdir -p {CONTROL}/wheels\ntar -xf /workspace/cdr214-wheels.tar -C {CONTROL}/wheels\n"
        install=f'python3 -m pip --python /opt/cdr214-venv install --no-index --find-links {CONTROL}/wheels numpy==2.5.0 scipy==1.18.1'
    body=f'''set -euo pipefail
trap 'echo $? > {CONTROL}/setup.exit' EXIT
echo '{packet['sha256']}  /workspace/cdr214-inputs.tar' | sha256sum -c -
mkdir -p {REMOTE}
tar -xf /workspace/cdr214-inputs.tar -C {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/cdr214-venv
{wheel_setup}{install}
/opt/cdr214-venv/bin/python -c "import torch,numpy,scipy; assert torch.__version__.split('+')[0]=='2.8.0'; assert torch.cuda.is_available(); print(torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name())"
/opt/cdr214-venv/bin/python -m pip freeze > {CONTROL}/environment-lock.txt
'''
    path=SCRATCH/(pod['name']+'-setup.sh');path.write_text(body,encoding='utf-8',newline='\n')
    copy(pod,path,CONTROL+'/setup.sh')
    ssh(pod,f'setsid bash {CONTROL}/setup.sh > {CONTROL}/setup.log 2>&1 < /dev/null &\necho $!\n')
    pod['setup_started_epoch']=time.time()


def launch(pod,phase):
    state=lease();assert time.time()<state['deadline_epoch']-1800 and spend(state)<32
    deadline=state.get('scientific_deadline_epoch',state['deadline_epoch']-1800)
    if phase=='calibration':
        command=f'python {REL}/00_profile.py --task {pod["task"]} --full --output {CONTROL}/profile.json\n'
        command+=f'python {REL}/02_worker.py --task {pod["task"]} --seed {pod["seed"]} --smoke --output {CONTROL}/short-worker --deadline-epoch {min(deadline,time.time()+1200)}\n'
    else:
        command=f'python {runrel(pod)}/02_worker.py --task {pod["task"]} --seed {pod["seed"]} --output {outpath(pod)} --deadline-epoch {deadline}\n'
    body=f'''set -euo pipefail
trap 'echo $? > {CONTROL}/{phase}.exit' EXIT
cd {REMOTE}
export PATH=/opt/cdr214-venv/bin:$PATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
{command}'''
    path=SCRATCH/(pod['name']+'-'+phase+'.sh');path.write_text(body,encoding='utf-8',newline='\n')
    copy(pod,path,CONTROL+'/'+phase+'.sh')
    pid=ssh(pod,f'set -euo pipefail\ntest ! -e {CONTROL}/{phase}.pid\nsetsid bash {CONTROL}/{phase}.sh > {CONTROL}/{phase}.log 2>&1 < /dev/null &\necho $! | tee {CONTROL}/{phase}.pid\n').strip()
    pod[phase+'_started_epoch']=time.time();pod[phase+'_pid']=pid


def status(pod):
    connect(pod)
    script='''python3 - <<'PY'
from pathlib import Path
import json,time,subprocess
c=Path(CONTROL);o=Path(OUTPUT);r={'epoch':time.time()}
for name in ('setup','calibration','production'):
 p=c/(name+'.exit')
 if p.exists():r[name+'_exit']=p.read_text().strip()
 p=c/(name+'.log')
 if p.exists():r[name+'_tail']=p.read_text()[-1200:]
p=c/'calibration216.exit'
if p.exists():r['calibration216_exit']=p.read_text().strip()
p=c/'profile216.json'
if p.exists():r['profile216']=json.loads(p.read_text())
p=c/'profile.json'
if p.exists():r['profile']=json.loads(p.read_text())
p=c/'microbatch.json'
if p.exists():r['microbatch']=json.loads(p.read_text())
p=c/'microbatch.exit'
if p.exists():r['microbatch_exit']=p.read_text().strip()
p=o/'worker_progress.json'
if p.exists():
 r['worker']=json.loads(p.read_text());p=o/r['worker']['job']['name']/'events.jsonl'
 if p.exists():
  lines=p.read_text().splitlines()
  if lines:r['event']=json.loads(lines[-1])
p=o/'worker_manifest.json'
if p.exists():r['worker_manifest']=json.loads(p.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
r['disk']=subprocess.check_output(['df','-h','/workspace'],text=True).strip()
print(json.dumps(r))
PY
'''.replace('CONTROL',repr(CONTROL)).replace('OUTPUT',repr(outpath(pod)))
    result=json.loads(ssh(pod,script));save(SCRATCH/(pod['name']+'-status.json'),result)
    return result


def runpath(pod):return ROOT/'runs/216-2026-11-02-sgd-transformer-microbatch512' if pod.get('run_id')=='216' else RUN
def runrel(pod):return runpath(pod).relative_to(ROOT).as_posix()
def outpath(pod):return f'{REMOTE}/{runrel(pod)}/artifacts/attempt-{pod["attempt"]:02d}/{pod["task"]}-seed{pod["seed"]}'


def retrieve(pod,phase):
    source=CONTROL if phase=='calibration' else outpath(pod)
    archive=SCRATCH/(pod['name']+'-'+phase+'.tar.gz')
    remote_archive='/workspace/cdr214-'+phase+'-outputs.tar.gz'
    script=f'''set -e
tar -I 'gzip -1' -cf {remote_archive} -C {source} .
sha256sum {remote_archive}
'''
    digest=ssh(pod,script,timeout=180).split()[0];begin=time.time()
    copy(pod,archive,remote_archive,False);assert file_hash(archive)==digest
    target=(HERE/'scratch/cloud-calibration'/pod['name']) if phase=='calibration' else (runpath(pod)/f'artifacts/attempt-{pod["attempt"]:02d}'/f'{pod["task"]}-seed{pod["seed"]}')
    assert not target.exists();target.mkdir(parents=True)
    with tarfile.open(archive) as bundle:bundle.extractall(target,filter='data')
    if phase=='production':
        inventory=read(target/'worker_inventory.json')
        for name,entry in inventory.items():
            path=target/name;assert path.resolve().is_relative_to(target.resolve())
            assert path.stat().st_size==entry['bytes'] and file_hash(path)==entry['sha256']
    receipt=dict(utc=stamp(),pod_id=pod['id'],task=pod['task'],seed=pod['seed'],phase=phase,
        archive_bytes=archive.stat().st_size,archive_sha256=digest,transfer_seconds=time.time()-begin,
        verified=True,path=target.relative_to(ROOT).as_posix())
    save(HERE/f'retrieval-{pod["name"]}-{phase}.json',receipt)
    return target


def delete(pod):
    if pod.get('production_pid') or pod.get('helper_pid'):
        receipt=read(HERE/('stream-retrieval-'+pod['name']+'.json'))
        acceptance=read(HERE/('worker-acceptance-'+pod['name']+'.json'))
        assert receipt['verified'] and acceptance['status'] in ('passed','partial_preserved')
        target=ROOT/receipt['path']
        assert file_hash(target/'worker_inventory.json')==receipt['inventory_sha256']==acceptance['inventory_sha256']
    elif pod.get('calibration_started_epoch') or pod.get('calibration216_started_epoch'):
        receipt=read(HERE/f'retrieval-{pod["name"]}-calibration.json')
        assert receipt['verified'] and (ROOT/receipt['path']).is_dir()
    owned(pod);cli('pod','delete',pod['id']);pod['deleted_epoch']=time.time()
    assert not any(x['id']==pod['id'] for x in cli('pod','list'))


def guard():
    while not (SCRATCH/'closed.json').exists():
        state=lease();now=time.time()
        if state['deadline_epoch']:
            soft=now>=state.get('scientific_deadline_epoch',state['deadline_epoch']-1800) or spend(state)>=32
            hard=now>=state['deadline_epoch'] or spend(state)>=38
            for pod in state['pods']:
                if pod.get('deleted_epoch'):continue
                try:
                    if soft and pod.get('host'):ssh(pod,f'touch {REMOTE}/{runrel(pod)}/STOP\n',timeout=20)
                    if hard:owned(pod);cli('pod','stop',pod['id'])
                except Exception as error:save(SCRATCH/'guard-error.json',dict(utc=stamp(),pod_id=pod['id'],error=repr(error)))
            if soft:save(SCRATCH/'stop-request.json',dict(utc=stamp(),estimated_usd=spend(state),hard=hard))
            if hard:return
        time.sleep(15)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','allocate','setup','calibration','production','status','retrieve-calibration','retrieve-production','delete','guard'])
    p.add_argument('--task',choices=['resnet','transformer']);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gpu',default='RTX 5090');p.add_argument('--cloud',default='SECURE',choices=['SECURE','COMMUNITY'])
    a=p.parse_args()
    if a.action=='guard':guard();return
    if a.action=='init':initialize();return
    if a.action=='allocate':allocate(a.task,a.seed,a.gpu,a.cloud);return
    state=lease();pod=next(x for x in reversed(state['pods']) if x['task']==a.task and x['seed']==a.seed and not x.get('deleted_epoch'))
    if a.action=='setup':setup(pod)
    elif a.action in ('calibration','production'):launch(pod,a.action)
    elif a.action=='status':
        result=status(pod)
        brief={k:v for k,v in result.items() if k not in ('profile','microbatch','setup_tail','calibration_tail','worker_manifest')}
        if 'profile' in result:brief['profile_status']=result['profile'].get('status','running')
        print(json.dumps(brief))
    elif a.action.startswith('retrieve-'):print(retrieve(pod,a.action.split('-')[1]))
    elif a.action=='delete':delete(pod)
    latest=lease();latest['pods']=[pod if x['id']==pod['id'] else x for x in latest['pods']];save(SCRATCH/'lease.json',latest)


if __name__=='__main__':main()
