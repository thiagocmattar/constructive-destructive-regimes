"""Task-owned H050 leases: bounded execution, verified Spark retention, teardown."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];S=HERE/'scratch/h050';S.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('transport',ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/03_cloud.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x);x.SCRATCH=S
REMOTE='/workspace/h050'
IMAGE='pytorch/pytorch@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be'
RUNS={289:'runs/289-2026-11-08-resnet-beta2-reductions',290:'runs/290-2026-11-08-transformer-beta2-reductions',291:'runs/291-2026-11-08-transformer-gradient-clipping'}
JOBS={'resnet':(289,['control','beta2_0p95','beta2_0p9','beta2_0p5']),
      'trans-control':(290,['control']),'trans-b95':(290,['beta2_0p95']),
      'trans-b90':(290,['beta2_0p9']),'trans-b50':(290,['beta2_0p5']),
      'trans-clip05':(291,['clip_0p5']),'trans-clip10':(291,['clip_1p0'])}
SPARK='tcml-spark@192.168.15.9'
OLD=json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
SPARKROOT=OLD['root'];SSH=['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','HostKeyAlias=spark-2277.local',SPARK]
SSH6=['ssh','-6','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','HostKeyAlias=spark-2277.local','tcml-spark@spark-2277.local']


def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n');temp.replace(p)


def record(p):dump(S/('lease-'+p['name']+'.json'),p)
def get(name):return json.loads((S/('lease-'+name+'.json')).read_text())
def emit(**d):print(json.dumps(dict(utc=x.stamp(),**d)),flush=True)


def spark(script,timeout=180):
    p=subprocess.run(SSH+['bash -s'],input=script.encode(),capture_output=True,timeout=timeout,creationflags=x.NO_WINDOW)
    # Retry only a failed connection, before any remote script could execute.
    # Never blindly retry a dropped or timed-out transfer with unknown remote state.
    if p.returncode and p.stderr.decode(errors='replace').startswith('ssh: connect to host 192.168.15.9 port 22:'):
        emit(event='spark_ipv6_connection_fallback')
        p=subprocess.run(SSH6+['bash -s'],input=script.encode(),capture_output=True,timeout=timeout,creationflags=x.NO_WINDOW)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
    return p.stdout.decode()


def launch(name):
    assert name in JOBS
    runid,arms=JOBS[name];run=RUNS[runid];packet=json.loads((ROOT/run/'packet.json').read_text())
    assert x.file_hash(ROOT/packet['path'])==packet['sha256']
    gpu=os.environ.get('H050_GPU','NVIDIA GeForce RTX 4090');cloud=os.environ.get('H050_CLOUD','SECURE')
    g=next(g for g in x.cli('gpu','list') if g['gpuId']==gpu)
    price=g['securePricePerHr'] if cloud=='SECURE' else g['communityPricePerHr']
    assert g['available'] and price is not None and price<=1.19
    dump(S/(name+'-catalog.json'),dict(utc=x.stamp(),gpu=g))
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    begin=time.time()
    if (S/('lease-'+name+'.json')).exists():
        pod=get(name);assert not pod.get('pid') and not pod.get('deleted_epoch')
    else:
        actual=x.create_wait('pod','create','--name','cdr-h050-'+name,'--image',IMAGE,'--gpu-id',g['gpuId'],
            '--gpu-count','1','--cloud-type',cloud,'--min-cuda-version','12.8','--container-disk-in-gb','30',
            '--volume-in-gb','30','--ports','22/tcp','--ssh','--env',json.dumps({'PUBLIC_KEY':public}),
            '--docker-args','bash -lc '+shlex.quote(bootstrap),'--wait','--wait-timeout','20s')
        pod=dict(name=name,id=actual['id'],created_epoch=begin,deadline_epoch=begin+7200,
            price_per_hour=actual.get('costPerHr',price),bounded_hourly_usd=1.22,
            run=run,arms=arms,output=run+'/artifacts/attempt-01/'+name,image=IMAGE,storage_gb=60)
        record(pod);emit(event='created',name=name,id=pod['id'])
    assert pod['price_per_hour']<=1.22
    for _ in range(60):
        try:
            actual=x.cli('pod','get',pod['id']);info=actual.get('ssh') or x.cli('ssh','info',pod['id'])
            if info.get('ip') and info.get('port'):
                pod.update(host=info['ip'],port=info['port']);record(pod)
                x.ssh(pod,'echo ready\n',timeout=20);break
        except Exception:pass
        time.sleep(6)
    else:raise RuntimeError('SSH unavailable; lease retained for watcher cleanup')
    x.ssh(pod,f'''set -e
mkdir -p {REMOTE}
python3 -m venv --system-site-packages --without-pip /opt/h050
python3 -m pip --python /opt/h050 install --no-cache-dir numpy==2.5.0 scipy==1.18.1 threadpoolctl==3.6.0 >/workspace/setup.log 2>&1
/opt/h050/bin/python -c "import torch; assert torch.__version__.split('+')[0]=='2.11.0'; assert torch.cuda.is_available()"
''',timeout=300)
    if (S/'peer_packets.json').exists() and runid in [290,291] and name!='trans-clip10':
        peer=json.loads((S/'peer_packets.json').read_text());packet=peer['packets'][str(runid)]
        x.ssh(pod,'printf "%s\\n" '+shlex.quote(peer['public_key'])+' >> /root/.ssh/authorized_keys\n')
        sender=get(peer['source'])
        command=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile=/workspace/h050-peer-hosts','-i','/root/h050-transfer-key',
            '-P',str(pod['port']),packet['path'],'root@'+pod['host']+':/workspace/inputs.tar.gz']
        x.ssh(sender,' '.join(shlex.quote(v) for v in command)+'\n',timeout=600)
    else:
        x.copy(pod,ROOT/packet['path'],'/workspace/inputs.tar.gz')
    preflight=x.ssh(pod,f'''set -e
/opt/h050/bin/python - <<'PY'
import hashlib,tarfile
from pathlib import Path
p=Path('/workspace/inputs.tar.gz')
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={packet['sha256']!r}
with tarfile.open(p) as t:t.extractall({REMOTE!r},filter='data')
PY
cd {REMOTE}
PYTHONPATH={REMOTE} /opt/h050/bin/python analyses/065-2026-11-07-mlp-frozen-lr-double/46_test_conventional.py
''',timeout=240)
    dump(ROOT/run/(name+'_preflight.json'),dict(status='passed',result=json.loads(preflight),packet_sha256=packet['sha256']))
    args=['/opt/h050/bin/python','-m','src.eos_tasks.conventional_continuation','--run',run,'--out',pod['output'],'--arms',*arms]
    script=f'''set -e
cd {REMOTE}
export PYTHONPATH={REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
set +e
remaining=$(({int(pod['deadline_epoch'])}-$(date +%s)-600))
timeout --signal=TERM --kill-after=30 "$remaining" {' '.join(shlex.quote(a) for a in args)} > /workspace/worker.log 2>&1
code=$?
echo "$code" > /workspace/worker.exit
if test "$code" = 0; then
 /opt/h050/bin/python analyses/065-2026-11-07-mlp-frozen-lr-double/47_verify_conventional.py {shlex.quote(pod['output'])} > /workspace/verification.log 2>&1
 echo "$?" > /workspace/verification.exit
fi
'''
    f=S/(name+'-worker.sh');f.write_text(script,encoding='utf-8',newline='\n');x.copy(pod,f,'/workspace/worker.sh')
    pid=x.ssh(pod,'setsid bash /workspace/worker.sh </dev/null >/workspace/driver.log 2>&1 &\necho $!\n').strip()
    pod.update(pid=pid,launched_utc=x.stamp());record(pod);dump(ROOT/run/(name+'_launch.json'),pod)
    emit(event='launched',name=name,arms=arms)


def status(name):
    p=get(name)
    if p.get('deleted_epoch'):return dict(name=name,status='deleted',verified=p.get('retrieval_verified'))
    if not p.get('host'):return dict(name=name,status='waiting_for_ssh')
    code=f'''python3 - <<'PY'
from pathlib import Path
import json
root=Path({REMOTE!r})/{p['output']!r};d={{'name':{name!r}}}
for key,path in [('exit',Path('/workspace/worker.exit')),('verification_exit',Path('/workspace/verification.exit'))]:
 d[key]=path.read_text().strip() if path.exists() else None
for key in ['progress','qualification','manifest']:
 f=root/(key+'.json')
 if f.exists():
  a=json.loads(f.read_text());d[key]=a if key!='manifest' else {{k:a[k] for k in ['status','error'] if k in a}}
print(json.dumps(d))
PY
'''
    return json.loads(x.ssh(p,code,timeout=30))


def collect(name,emergency=False):
    p=get(name);out=p['output']
    state=status(name)
    if not emergency:assert state.get('exit')=='0' and state.get('verification_exit')=='0',state
    # Pull directly from owned Spark; the temporary private key never leaves Spark.
    key=SPARKROOT+'/analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h050-retention-key'
    public=spark(f'''mkdir -p {shlex.quote(key.rsplit('/',1)[0])}
if ! test -f {shlex.quote(key)}; then ssh-keygen -q -t ed25519 -N '' -f {shlex.quote(key)}; fi
chmod 600 {shlex.quote(key)}
cat {shlex.quote(key+'.pub')}
''').strip()
    x.ssh(p,'printf "%s\\n" '+shlex.quote(public)+' >> /root/.ssh/authorized_keys\n')
    audit=HERE/'52_check_original_fork.py';rel=audit.relative_to(ROOT).as_posix()
    x.copy(p,audit,REMOTE+'/'+rel)
    if not emergency:
        x.ssh(p,f'cd {REMOTE}\n/opt/h050/bin/python {shlex.quote(rel)} --run {shlex.quote(p["run"])} --out {shlex.quote(out)} --report-name complete_prefix_audit.json\n',timeout=180)
    transfer=json.loads(x.ssh(p,f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil
out=Path({REMOTE!r})/{out!r}
logs=out/'execution_logs';logs.mkdir(exist_ok=True)
for name in ['worker.log','verification.log','extended-supervisor.log']:
 source=Path('/workspace')/name
 if source.exists():shutil.copy2(source,logs/name)
files=[]
for source in sorted(out.rglob('*')):
 if source.is_file() and source.name!='transfer_inventory.json':
  with source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=source.relative_to(out).as_posix(),bytes=source.stat().st_size,sha256=digest))
(out/'transfer_inventory.json').write_text(json.dumps(dict(files=files,posthoc_verifier_sha256={x.file_hash(audit)!r})))
print(json.dumps(files))
PY
''',timeout=180))
    needed=json.loads(spark(f'''python3 - <<'PY'
import hashlib,json
from pathlib import Path
root=Path({(SPARKROOT+'/'+out)!r});needed=[]
for item in {transfer!r}:
 path=root/item['path'];matches=False
 if path.is_file() and path.stat().st_size==item['bytes']:
  with path.open('rb') as f:matches=hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
 if not matches:needed.append(item['path'])
needed.append('transfer_inventory.json')
print(json.dumps(needed))
PY
''',timeout=180))
    paths=' '.join(shlex.quote(out+'/'+name) for name in needed)
    remote_ssh=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+key+'.hosts','-i',key,
        '-p',str(p['port']),'root@'+p['host'],'bash -o pipefail -c '+shlex.quote(f'tar -C {shlex.quote(REMOTE)} -cf - {paths} | gzip -1')]
    spark(f'''python3 - <<'PY'
import subprocess
from pathlib import Path
root=Path({SPARKROOT!r});(root/{out!r}).mkdir(parents=True,exist_ok=True)
producer=subprocess.Popen({remote_ssh!r},stdout=subprocess.PIPE,stderr=subprocess.PIPE)
consumer=subprocess.Popen(['tar','-C',str(root),'-xzf','-'],stdin=producer.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
producer.stdout.close();_,error=consumer.communicate(timeout=1200);producer.wait(timeout=60)
assert consumer.returncode==0 and producer.returncode==0,error.decode()
print('Direct Spark retention transfer completed')
PY
''',timeout=1280)
    verified=json.loads(spark(f'''python3 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path({(SPARKROOT+'/'+out)!r});m=json.loads((p/'transfer_inventory.json').read_text())
for f in m['files']:
 q=p/f['path']
 with q.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
 assert q.stat().st_size==f['bytes'] and digest==f['sha256'],str(q)
print(json.dumps(dict(status='passed',files=len(m['files']),bytes=sum(f['bytes'] for f in m['files']),root=str(p))))
PY
''',timeout=300))
    receipt=json.loads(x.ssh(p,f'''python3 - <<'PY'
import json,tarfile,hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});out=root/{out!r}
for source in ['worker.log','verification.log']:
 p=Path('/workspace')/source
 if p.exists():shutil.copy2(p,out/source)
files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log']]
archive=Path('/workspace/compact.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for f in files:t.add(f,arcname=f.relative_to(root).as_posix())
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=digest,files=len(files))))
PY
''',timeout=90))
    archive=S/(name+'-compact.tar.gz');x.copy(p,archive,'/workspace/compact.tar.gz',upload=False)
    assert archive.stat().st_size==receipt['bytes'] and x.file_hash(archive)==receipt['sha256']
    with tarfile.open(archive) as t:t.extractall(ROOT,filter='data')
    for f in json.loads((ROOT/out/'transfer_inventory.json').read_text())['files']:
        q=ROOT/out/f['path']
        if q.exists():assert q.stat().st_size==f['bytes'] and x.file_hash(q)==f['sha256']
    p.update(retrieval_verified=True,retained_spark=verified,compact=receipt);record(p)
    dump(ROOT/p['run']/(name+'_retrieval.json'),dict(**verified,compact=receipt,verified=True,remote_inventory=out+'/transfer_inventory.json'))
    x.cli('pod','delete',p['id']);p['deleted_epoch']=time.time();record(p)
    dump(ROOT/p['run']/(name+'_closeout.json'),p);emit(event='verified_and_deleted',name=name,bytes=verified['bytes'])


def watch():
    pending={}
    with ThreadPoolExecutor(max_workers=7) as pool:
        while not (S/'STOP_WATCHER').exists():
            now=time.time();spent=0.
            for f in list(S.glob('lease-*.json'))+list(S.glob('aborted-*.json')):
                p=json.loads(f.read_text());spent+=(p.get('deleted_epoch',p.get('stopped_epoch',now))-p['created_epoch'])/3600*p['bounded_hourly_usd']
            for f in S.glob('lease-*.json'):
                p=json.loads(f.read_text());name=p['name']
                if p.get('deleted_epoch') or p.get('stopped_epoch'):continue
                try:
                    if spent>=19.5:
                        x.cli('pod','stop',p['id']);p['stopped_epoch']=time.time();record(p)
                        emit(event='budget_guard_stopped_gpu',name=name,conservative_spend=spent);continue
                    if name in pending:
                        if not pending[name].done():continue
                        future=pending.pop(name)
                        try:future.result()
                        except Exception as error:emit(event='retrieval_error',name=name,error=str(error)[-1000:])
                        continue
                    st=status(name)
                    emit(event='status',**{k:v for k,v in st.items() if k!='qualification'})
                    if st.get('exit')=='0' and st.get('verification_exit')=='0':
                        stage_policy=S/'prestage.json'
                        requires_receipt=stage_policy.exists() and name in json.loads(stage_policy.read_text()).get('workers',[])
                        missing_receipt=requires_receipt and not (ROOT/p['run']/(name+'_immutable_prestage.json')).exists()
                        if (S/('prestage-active-'+name+'.json')).exists() or missing_receipt:
                            emit(event='waiting_for_immutable_prestage',name=name)
                        else:pending[name]=pool.submit(collect,name)
                    elif time.time()>p['deadline_epoch']-480:
                        if p.get('host'):
                            x.ssh(p,'pkill -TERM -f "python -m src.eos_tasks.conventional_continuation" || true\n',timeout=30)
                            time.sleep(35)
                            try:collect(name,emergency=True);continue
                            except Exception as error:emit(event='emergency_retrieval_error',name=name,error=str(error))
                        x.cli('pod','stop',p['id']);p['stopped_epoch']=time.time();record(p)
                except Exception as error:emit(event='watch_error',name=name,error=str(error)[-600:])
            time.sleep(30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','status','collect','watch']);parser.add_argument('names',nargs='*');a=parser.parse_args()
    if a.action=='watch':watch();return
    if a.action=='launch':
        if not (S/'watcher.json').exists():
            with (S/'watcher.log').open('xb') as f:
                p=subprocess.Popen([sys.executable,str(Path(__file__)),'watch'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=x.NO_WINDOW)
            dump(S/'watcher.json',dict(pid=p.pid,started_utc=x.stamp()))
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures={pool.submit(launch,n):n for n in a.names}
            for f in as_completed(futures):
                try:f.result()
                except Exception as error:emit(event='launch_failed',name=futures[f],error=str(error)[-1500:])
    elif a.action=='status':
        with ThreadPoolExecutor(max_workers=7) as pool:
            for d in pool.map(status,a.names or [p.stem[6:] for p in S.glob('lease-*.json')]):print(json.dumps(d),flush=True)
    else:
        for n in a.names:collect(n)


if __name__=='__main__':main()
