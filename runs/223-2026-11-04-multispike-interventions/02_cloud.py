"""Run223 task-owned controller, independent USD15 guard, verified collection."""
import argparse,concurrent.futures,hashlib,importlib.util,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('transport215',ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py');x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
x.HERE=x.RUN=HERE;x.SCRATCH=HERE/'scratch';x.REMOTE='/workspace/cdr223';x.CONTROL='/workspace/cdr223-control';x.PREFIX='cdr223-';x.SCRATCH.mkdir(exist_ok=True)
S=x.SCRATCH;REMOTE=x.REMOTE;REL=HERE.relative_to(ROOT).as_posix();OUT=REMOTE+'/'+REL+'/artifacts/worker';write=x.write_json

def state():return json.loads((S/'production-lease.json').read_text())
def persist(st):write(S/'production-lease.json',st)
def cost(st):return x.spend(st)
def log(**v):print(json.dumps(dict(utc=x.stamp(),**v)),flush=True)

def bundle(case):
    manifest=json.loads((HERE/'input_manifest.json').read_text());p=S/(case+'-inputs.tar.gz')
    with tarfile.open(p,'w:gz',compresslevel=1) as t:
        for f in manifest['files']:
            if case in f['tasks']:t.add(ROOT/f['path'],arcname=f['path'])
        t.add(HERE/'input_manifest.json',arcname=REL+'/input_manifest.json')
    return p

def newpod(st,case,role,gpu=None):
    assert sum(not p.get('deleted_epoch') for p in st['pods'])<3
    if cost(st)>=10:raise RuntimeError('Budget admission closed before another Pod.')
    catalog=x.cli('gpu','list','--include-unavailable')
    preferred=['RTX 4090','RTX 5090','RTX PRO 4500 SE','RTX PRO 4500','RTX 6000 Ada','RTX PRO 4000','L40S','A100 SXM','A100 PCIe']
    candidates=[g for name in preferred for g in catalog if g['displayName']==name and g.get('available') and g.get('securePricePerHr') and g['securePricePerHr']<=1.79 and (not gpu or g['gpuId']==gpu)]
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    for g in candidates:
        name='cdr223-'+case+'-'+role+'-'+str(len(st['pods']))
        try:
            started=time.time()
            actual=x.create_wait('pod','create','--name',name,'--image',st['image'],'--gpu-id',g['gpuId'],'--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0','--container-disk-in-gb','60','--volume-in-gb','20','--ports','22/tcp','--ssh','--env',json.dumps({'PUBLIC_KEY':public}),'--docker-args','bash -lc '+shlex.quote(bootstrap),'--wait','--wait-timeout','20s')
        except Exception as e:
            log(event='capacity_unavailable',gpu=g['displayName'],error=str(e)[:300]);continue
        pod=dict(id=actual['id'],name=name,case=case,role=role,gpu=g['displayName'],gpu_id=g['gpuId'],price_per_gpu_hour=float(actual.get('costPerHr') or g['securePricePerHr']),created_epoch=started)
        st['pods'].append(pod);persist(st);write(S/(pod['id']+'-creation.json'),actual);log(event='created',pod=pod)
        return pod
    return None

def prepare(pod,packet):
    deadline=time.time()+480
    while time.time()<deadline:
        try:
            actual=x.cli('pod','get',pod['id']);info=actual.get('ssh') or x.cli('ssh','info',pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'],port=info['port']);x.ssh(pod,'echo ready\n',timeout=20);break
        except Exception:pass
        time.sleep(8)
    else:raise RuntimeError('SSH startup exceeded8minutes')
    x.ssh(pod,'mkdir -p '+x.CONTROL+'\n');x.copy(pod,packet,'/workspace/input223.tar.gz')
    setup=f'''set -euo pipefail
mkdir -p {REMOTE}
echo '{x.file_hash(packet)}  /workspace/input223.tar.gz' | sha256sum -c -
tar -xzf /workspace/input223.tar.gz -C {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/cdr223-venv
python3 -m pip --python /opt/cdr223-venv install --no-cache-dir --disable-pip-version-check numpy==2.5.0 scipy==1.18.1
/opt/cdr223-venv/bin/python -c "import torch; assert torch.__version__.split('+')[0]=='2.11.0'; assert torch.cuda.is_available();print(torch.__version__,torch.cuda.get_device_name())"
'''
    f=S/(pod['id']+'-setup.sh');f.write_text(setup,newline='\n');x.copy(pod,f,x.CONTROL+'/setup.sh')
    driver=S/(pod['id']+'-setup-driver.sh');driver.write_text(f'bash {x.CONTROL}/setup.sh > {x.CONTROL}/setup.log 2>&1\necho $? > {x.CONTROL}/setup.exit\n',newline='\n');x.copy(pod,driver,x.CONTROL+'/setup-driver.sh')
    x.ssh(pod,f'setsid bash {x.CONTROL}/setup-driver.sh </dev/null > {x.CONTROL}/driver.log 2>&1 &\necho $!\n')
    deadline=time.time()+600
    while time.time()<deadline:
        r=x.ssh(pod,f'if test -f {x.CONTROL}/setup.exit; then cat {x.CONTROL}/setup.exit; else echo WAIT; fi\n').strip()
        if r!='WAIT':
            if r!='0':raise RuntimeError('Setup failed: '+x.ssh(pod,f'tail -c 2000 {x.CONTROL}/setup.log\n'))
            return
        time.sleep(10)
    raise RuntimeError('Setup timeout')

def launch(pod,mode,arms='',parent=None,reference=None):
    folder=OUT if mode!='calibrate' else REMOTE+'/'+REL+'/artifacts/calibration'
    command=f'/opt/cdr223-venv/bin/python -u {REL}/01_worker.py --case {pod["case"]} --mode {mode} --output {folder} --seconds 21600 --arms {shlex.quote(arms)}'
    if parent:command+=' --parent '+shlex.quote(parent)+' --reference '+shlex.quote(reference)
    script=f'''cd {REMOTE}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
{command}
echo $? > {x.CONTROL}/{mode}.exit
'''
    f=S/(pod['id']+'-'+mode+'.sh');f.write_text(script,newline='\n');x.copy(pod,f,x.CONTROL+'/'+mode+'.sh')
    pid=x.ssh(pod,f'setsid bash {x.CONTROL}/{mode}.sh > {x.CONTROL}/{mode}.log 2>&1 </dev/null &\necho $!\n').strip()
    pod['mode']=mode;pod['pid']=pid;log(event='launched',case=pod['case'],role=pod['role'],mode=mode,pid=pid)

def readjson(pod,path):
    return json.loads(x.ssh(pod,"python3 -c "+shlex.quote("from pathlib import Path; p=Path("+repr(path)+"); print(p.read_text() if p.exists() else '{}')")+'\n'))

def calibration(pod):
    launch(pod,'calibrate')
    until=time.time()+240
    while time.time()<until:
        r=readjson(pod,REMOTE+'/'+REL+'/artifacts/calibration/manifest.json')
        if r.get('status') in ('completed','failed'):
            if r['status']!='completed':raise RuntimeError(str(r))
            seconds=max(r['seconds'][1:]);pod['seconds_per_update']=seconds;log(event='calibrated',pod=pod['id'],seconds=seconds)
            cfg=json.loads((HERE/'config.json').read_text())['cases'][pod['case']]
            steps=cfg['start']+2*(cfg['end']-cfg['start'])+16
            predicted=seconds*steps*1.2/3600*pod['price_per_gpu_hour']
            assert predicted<=(9.0 if pod['case']=='transformer' else 2.8),(predicted,'Admission cost too high')
            return
        time.sleep(8)
    raise RuntimeError('Calibration timeout')

def collect(pod,terminal=True):
    script=f'''set -e
mkdir -p {REMOTE}/{REL}/artifacts/control
cp -a {x.CONTROL}/. {REMOTE}/{REL}/artifacts/control/
python3 - <<'PY'
import pathlib,json,hashlib,tarfile
p=pathlib.Path({(REMOTE+'/'+REL+'/artifacts')!r}); rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='transfer_inventory.json':
  with f.open('rb') as s:h=hashlib.file_digest(s,'sha256').hexdigest()
  rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h))
(p/'transfer_inventory.json').write_text(json.dumps(dict(files=rows)))
with tarfile.open('/workspace/output223.tar.gz','w:gz',compresslevel=1) as t:t.add(p,arcname='payload')
with open('/workspace/output223.tar.gz','rb') as s:print(hashlib.file_digest(s,'sha256').hexdigest())
PY
'''
    digest=x.ssh(pod,script,timeout=900).strip().splitlines()[-1]
    folder=HERE/'artifacts'/pod['id'];folder.mkdir(parents=True,exist_ok=True);archive=folder/'output.tar.gz'
    x.copy(pod,archive,'/workspace/output223.tar.gz',upload=False);assert x.file_hash(archive)==digest
    target=Path('\\\\?\\'+str(folder.resolve())) if os.name=='nt' else folder
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    rows=json.loads((target/'payload/transfer_inventory.json').read_text())['files']
    for r in rows:
        f=target/'payload'/r['path'];assert f.resolve().is_relative_to((target/'payload').resolve());assert f.stat().st_size==r['bytes'] and x.file_hash(f)==r['sha256']
    receipt=dict(pod_id=pod['id'],verified=True,files=len(rows),archive_sha256=digest,utc=x.stamp());write(folder/'receipt.json',receipt);pod['retrieved']=True
    return receipt

def retire(pod):
    actual=x.cli('pod','get',pod['id']);assert actual['name']==pod['name'] and pod['name'].startswith('cdr223-')
    if pod.get('host'):
        try:collect(pod)
        except Exception as e:
            log(event='collection_failed',pod=pod['id'],error=repr(e));x.cli('pod','stop',pod['id']);pod['stopped']=True;return
    x.cli('pod','delete',pod['id']);pod['deleted_epoch']=time.time();log(event='deleted',pod=pod['id'])

def main():
    assert not (S/'production-lease.json').exists(),'Use monitor/recovery, never overwrite an existing lease.'
    cfg=json.loads((HERE/'config.json').read_text());image=json.loads((ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/scratch/image-identity.json').read_text())['image']
    st=dict(run_id='223',created_utc=x.stamp(),deadline_epoch=time.time()+8*3600,ceiling_usd=15,soft_stop_usd=12,compute_stop_usd=14,image=image,pods=[])
    persist(st)
    with (S/'guard.log').open('w') as f:subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'guard'],stdout=f,stderr=subprocess.STDOUT,creationflags=x.NO_WINDOW)
    packets={c:bundle(c) for c in ['resnet','transformer']}
    # Acquire only the two leaders first; the third GPU is admitted at the shared fork.
    for case in ['transformer','resnet']:
        pod=newpod(st,case,'leader');persist(st)
        if pod is None:continue
        try:prepare(pod,packets[case]);calibration(pod);launch(pod,'leader','fixed_half' if case=='transformer' else 'fixed_half,cosine32_half')
        except Exception as e:pod['setup_error']=repr(e);log(event='failed_setup',pod=pod['id'],error=repr(e));retire(pod)
        persist(st)
    monitor(st,packets)

def monitor(st,packets):
    for tombstone in S.glob('*-retired.json'):
        retired=json.loads(tombstone.read_text())
        for pod in st['pods']:
            if pod['id']==retired['id']:pod.update(retired)
    persist(st)
    while time.time()<st['deadline_epoch'] and cost(st)<14:
        active=[p for p in st['pods'] if not p.get('deleted_epoch') and not p.get('stopped') and p.get('mode')=='leader']
        # Retry missing leader capacity without changing science or exceeding budget.
        for case in ['transformer','resnet']:
            if not any(p['case']==case and not p.get('setup_error') and (p.get('mode')=='leader' or (p.get('mode') is None and not p.get('deleted_epoch'))) for p in st['pods']) and cost(st)<4:
                pod=newpod(st,case,'leader');persist(st)
                if pod:
                    try:prepare(pod,packets[case]);calibration(pod);launch(pod,'leader','fixed_half' if case=='transformer' else 'fixed_half,cosine32_half')
                    except Exception as e:pod['setup_error']=repr(e);retire(pod)
                    persist(st)
        for pod in list(st['pods']):
            if pod.get('deleted_epoch') or pod.get('stopped') or pod.get('setup_error') or pod.get('mode') not in ('leader','follower'):continue
            try:
                manifest=readjson(pod,OUT+'/manifest.json');pod['last_manifest']=manifest
                if pod['role']=='follower':
                    q=readjson(pod,OUT+'/cross_worker_qualification.json');pod['cross_worker_qualified']=q.get('status')=='passed'
                ready=readjson(pod,OUT+'/READY.json') if pod['role']=='leader' and pod['case']=='transformer' else {}
                if ready and not pod.get('follower_dispatched'):
                    follower=newpod(st,'transformer','follower',gpu=pod['gpu_id']);persist(st)
                    if follower:
                        try:
                            prepare(follower,packets['transformer'])
                            for name,remote,expected in [('fork',OUT+'/prefix/end.pt',ready['parent_sha256']),('reference',OUT+'/qualification/direct/end.pt',ready['reference_sha256'])]:
                                local=S/(pod['id']+'-'+name+'.pt');x.copy(pod,local,remote,upload=False);assert x.file_hash(local)==expected
                                x.copy(follower,local,'/workspace/'+name+'.pt');actual=x.ssh(follower,'sha256sum /workspace/'+name+'.pt\n').split()[0];assert actual==expected
                            launch(follower,'follower','cosine32_half','/workspace/fork.pt','/workspace/reference.pt');pod['follower_dispatched']=True
                        except Exception as e:follower['setup_error']=repr(e);retire(follower)
                        persist(st)
                if manifest.get('status') in ('completed','failed'):
                    if pod['case']=='transformer' and pod['role']=='leader' and manifest['status']=='completed' and not pod.get('follower_dispatched'):continue
                    if pod['case']=='transformer' and pod['role']=='leader' and manifest['status']=='completed' and not any(q.get('role')=='follower' and q.get('cross_worker_qualified') for q in st['pods']):
                        continue
                    pod['terminal_status']=manifest['status'];retire(pod);persist(st)
            except Exception as e:pod['last_monitor_error']=repr(e);log(event='monitor_error',pod=pod['id'],error=repr(e))
        persist(st);write(HERE/'progress.json',dict(utc=x.stamp(),estimated_usd=cost(st),pods=st['pods']))
        completed=[p for p in st['pods'] if p.get('terminal_status')=='completed' and p.get('retrieved')]
        if any(p['case']=='resnet' for p in completed) and sum(p['case']=='transformer' for p in completed)>=2:
            write(S/'production-closed.json',dict(status='completed',estimated_usd=cost(st),utc=x.stamp()));return
        time.sleep(30)
    write(HERE/'needs_attention.json',dict(status='budget_or_deadline',estimated_usd=cost(st),utc=x.stamp()))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='guard':x.guard()
    elif len(sys.argv)>1 and sys.argv[1]=='monitor':monitor(state(),{c:S/(c+'-inputs.tar.gz') for c in ['resnet','transformer']})
    else:main()
