"""Detached post-run replay and hash-verified retrieval for H049 (owned Spark)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepare',HERE/'36_prepare_cap_lr_sweep.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def postprocess():
    receipt=HERE/'cap_lr_audit_launch.json';assert not receipt.exists()
    script=HERE/'37_audit_cap_lr.py';rel=script.relative_to(x.ROOT).as_posix()
    subprocess.run(x.SCP+[str(script),x.HOST+':'+x.REMOTE+'/'+rel],check=True,timeout=180)
    driver=f'''import json,time,subprocess,os,hashlib
from pathlib import Path
root=Path({x.REMOTE!r});start=time.monotonic()
script=root/{rel!r}
assert hashlib.sha256(script.read_bytes()).hexdigest()=={x.sha(script)!r}
status=root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h049-attempt-01-status.json'
while True:
 if status.exists():
  data=json.loads(status.read_text())
  assert all(r['returncode']==0 for r in data['results']),data
  if len(data['results'])==3:break
 assert time.monotonic()-start<5500,'Training did not finish'
 time.sleep(30)
env=dict(os.environ,PYTHONPATH={x.OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
results=[]
for rel in {[r.relative_to(x.ROOT).as_posix() for r in x.RUNS]!r}:
 run=root/rel
 with (run/'scratch/cap-lr-replay.log').open('xb') as stream:
  p=subprocess.run([{x.OLD['python']!r},str(script),'--run',str(run),'--worker',str(run/'artifacts/attempt-01/worker')],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
 results.append(dict(run=rel,returncode=p.returncode))
 (root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch/h049-replay-status.json').write_text(json.dumps(dict(results=results)))
 if p.returncode:raise SystemExit(p.returncode)
'''
    result=x.remote(f'''import subprocess,json,datetime
from pathlib import Path
scratch=Path({x.REMOTE!r})/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch'
driver=scratch/'h049-postprocess.py';assert not driver.exists();driver.write_text({driver!r})
with (scratch/'h049-postprocess.log').open('xb') as stream:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30','8300','python3',str(driver)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_sha256={x.sha(script)!r},training_unchanged=True)))''')
    x.dump(receipt,result);print(json.dumps(result))


def status():
    result=x.remote(f'''import json,subprocess
from pathlib import Path
root=Path({x.REMOTE!r});workers=[];total=0
for rel in {[r.relative_to(x.ROOT).as_posix() for r in x.RUNS]!r}:
 run=root/rel;w=run/'artifacts/attempt-01/worker';item=dict(run=run.name)
 if not w.exists():item['status']='pending';workers.append(item);continue
 if (w/'manifest.json').exists():
  m=json.loads((w/'manifest.json').read_text());item.update(status=m['status'],complete=len(m['results']),failed=[r['arm']['id'] for r in m['results'] if r['status']!='completed']);total+=len(m['results'])
 p=json.loads((w/'progress.json').read_text());arm=p.get('arm')
 if arm and (w/arm/'progress.json').exists():item.update(current=json.loads((w/arm/'progress.json').read_text()))
 if (w/'cap_lr_replay.json').exists():item['replay']=json.loads((w/'cap_lr_replay.json').read_text())['status']
 item['raw_bytes']=sum(f.stat().st_size for f in w.rglob('*') if f.is_file())
 workers.append(item)
scratch=root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch'
extra={{}}
for name in ['h049-attempt-01-status.json','h049-replay-status.json']:
 if (scratch/name).exists():extra[name]=json.loads((scratch/name).read_text())
print(json.dumps(dict(completed=total,total=54,workers=workers,driver=extra,gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout)))''')
    x.dump(HERE/'scratch/cap-lr-current-status.json',result)
    print(json.dumps(dict(completed=result['completed'],total=54,gpu=result['gpu'].strip(),
        workers=[dict(run=w['run'],status=w.get('status','starting'),complete=w.get('complete',0),
            active=None if w.get('status')=='completed' else w.get('current'),replay=w.get('replay'),failed=w.get('failed',[])) for w in result['workers']],
        raw_bytes=sum(w.get('raw_bytes',0) for w in result['workers']),driver=result['driver'])))


def collect(seed):
    run=x.RUNS[seed];rel=run.relative_to(x.ROOT).as_posix();receipt_path=run/'spark_retrieval.json'
    assert not receipt_path.exists(),'Already retrieved'
    receipt=x.remote(f'''import json,hashlib,tarfile
from pathlib import Path
run=Path({x.REMOTE!r})/{rel!r};root=run/'artifacts';w=root/'attempt-01/worker'
assert json.loads((w/'manifest.json').read_text())['status']=='completed'
assert json.loads((w/'cap_lr_replay.json').read_text())['status']=='passed'
logs=root/'execution_logs';logs.mkdir(exist_ok=True)
for name in ['attempt-01.log','mechanics-01.log','cap-lr-replay.log']:
 source=run/'scratch'/name
 if source.exists():(logs/name).write_bytes(source.read_bytes())
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p!=root/'transfer_inventory.json':
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
inventory=root/'transfer_inventory.json';inventory.write_text(json.dumps(dict(files=files)))
archive=run/'scratch/h049-results.tar'
with tarfile.open(archive,'w') as tar:
 for f in files:tar.add(root/f['path'],arcname=f['path'])
 tar.add(inventory,arcname='transfer_inventory.json')
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(files),raw_bytes=sum(f['bytes'] for f in files))))''',timeout=600)
    target=run/'artifacts';target.mkdir(parents=True,exist_ok=True)
    # Low local free space: NTFS compression preserves logical bytes, and a
    # hashing SSH stream avoids retaining a second local archive during extraction.
    assert target.resolve().is_relative_to(run.resolve())
    if os.name=='nt':subprocess.run(['compact','/C','/I',str(target)],check=True,capture_output=True)
    class HashingStream:
        def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.size=0
        def read(self,n=-1):
            b=self.stream.read(n);self.digest.update(b);self.size+=len(b);return b
    command=x.SSH[:-1]+['-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4',x.HOST,'cat','--',receipt['archive']]
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as process:
        stream=HashingStream(process.stdout)
        with tarfile.open(fileobj=stream,mode='r|') as tar:
            for member in tar:
                assert member.isfile(),member.name
                tar.extract(member,target,filter='data')
                if os.name=='nt' and member.size>1024*1024:
                    p=(target/member.name).resolve();assert p.is_relative_to(target.resolve())
                    subprocess.run(['compact','/C','/I','/F',str(p)],check=True,capture_output=True)
        while stream.read(1024*1024):pass
        stderr=process.stderr.read();assert process.wait(timeout=60)==0,stderr
    assert stream.size==receipt['bytes'] and stream.digest.hexdigest()==receipt['sha256']
    for f in json.loads((target/'transfer_inventory.json').read_text())['files']:
        p=target/f['path'];assert p.stat().st_size==f['bytes'] and x.sha(p)==f['sha256'],str(p)
    # Only remove our verified redundant transport archive; raw evidence remains.
    cleanup=x.remote(f'''from pathlib import Path
import json
p=Path({receipt['archive']!r});assert p.resolve().is_relative_to((Path({x.REMOTE!r})/{rel!r}/'scratch').resolve())
p.unlink();print(json.dumps(dict(transport_archive_removed=True)))''')
    receipt.update(verified=True,local=target.relative_to(x.ROOT).as_posix(),local_archive_not_materialized=True,
        retrieval='Native SSH stream; complete archive digest plus per-file size/SHA-256; NTFS compression on Windows',**cleanup)
    x.dump(receipt_path,receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['postprocess','status','collect']);p.add_argument('--seed',type=int,choices=[0,1,2]);a=p.parse_args()
    if a.action=='postprocess':postprocess()
    elif a.action=='status':status()
    else:assert a.seed is not None;collect(a.seed)
