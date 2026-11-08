"""Stage, smoke, launch and monitor the prespecified H049 sweep on owned Spark."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS=[ROOT/f'runs/{n}-2026-11-08-mlp-cap-lr-seed{s}' for s,n in enumerate([286,287,288])]
OLD=json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
REMOTE=OLD['root'];HOST='tcml-spark@spark-2277'
SSH=['ssh','-4','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','HostKeyAlias=spark-2277.local',HOST]
SCP=['scp','-4','-q','-o','BatchMode=yes','-o','HostKeyAlias=spark-2277.local']

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def remote(code,timeout=120):
    p=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=timeout)
    assert p.returncode==0,p.stderr+'\n'+p.stdout
    return json.loads(p.stdout)

def stage():
    prior=json.loads((ROOT/'runs/272-2026-11-07-mlp-displacement-confirmation/input_manifest.json').read_text())
    shared=[]
    for f in prior['files']:
        if f['path'].startswith('runs/272-'):continue
        p=ROOT/f.get('local_path',f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
        shared.append(f)
    all_files={}
    for run in RUNS:
        cfg=json.loads((run/'config.json').read_text());files={f['path']:f for f in shared}
        extras=[run/'config.json',run/'01_worker.py',RUNS[0]/'02_test.py',
                ROOT/'src/eos_tasks/distributed_cap_sweep.py',ROOT/cfg['initial_state']]
        for p in extras:
            name=p.relative_to(ROOT).as_posix();files[name]=dict(path=name,bytes=p.stat().st_size,sha256=sha(p))
        manifest=dict(case='mlp',seed=cfg['seed'],files=list(files.values()))
        target=run/'input_manifest.json'
        if target.exists():assert json.loads(target.read_text())==manifest,'Never overwrite frozen inputs'
        else:dump(target,manifest)
        all_files.update(files)
        name=target.relative_to(ROOT).as_posix();all_files[name]=dict(path=name,bytes=target.stat().st_size,sha256=sha(target))
    missing=remote(f'''from pathlib import Path
import hashlib,json
root=Path({REMOTE!r}); missing=[]
for f in {list(all_files.values())!r}:
 p=root/f['path']
 if not p.exists():missing.append(f['path'])
 else:
  with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
  assert p.stat().st_size==f['bytes'] and digest==f['sha256'],str(p)
print(json.dumps(missing))''')
    packet=HERE/'scratch/cap-lr-sweep-inputs.tar.gz';packet.parent.mkdir(exist_ok=True)
    with tarfile.open(packet,'w:gz',compresslevel=1) as tar:
        for name in missing:
            f=all_files[name];tar.add(ROOT/f.get('local_path',name),arcname=name)
    dest=REMOTE+'/h049-inputs.tar.gz'
    subprocess.run(SCP+[str(packet),HOST+':'+dest],check=True,timeout=300)
    result=remote(f'''from pathlib import Path
import hashlib,json,tarfile,subprocess,os
root=Path({REMOTE!r}); archive=Path({dest!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(packet)!r}
with tarfile.open(archive) as t:t.extractall(root,filter='data')
for f in {list(all_files.values())!r}:
 p=root/f['path']
 with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
 assert p.stat().st_size==f['bytes'] and digest==f['sha256'],str(p)
env=dict(os.environ,PYTHONPATH={OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8')
test=subprocess.run([{OLD['python']!r},str(root/{(RUNS[0]/'02_test.py').relative_to(ROOT).as_posix()!r})],env=env,cwd=root,capture_output=True,text=True)
assert test.returncode==0,test.stderr
print(json.dumps(dict(status='passed',uploaded={len(missing)},files={len(all_files)},tests=json.loads(test.stdout))))''',timeout=180)
    dump(HERE/'cap_lr_preflight.json',result);print(json.dumps(result))

def launch(mechanics):
    label='mechanics-01' if mechanics else 'attempt-01'
    receipt=HERE/('cap_lr_smoke_launch.json' if mechanics else 'cap_lr_launch.json')
    assert not receipt.exists(),'Use a new attempt for retries'
    runs=RUNS[:1] if mechanics else RUNS
    specs=[dict(run=r.relative_to(ROOT).as_posix(),worker=(r/'01_worker.py').relative_to(ROOT).as_posix()) for r in runs]
    driver=f'''import json,os,subprocess,time,datetime
from pathlib import Path
root=Path({REMOTE!r});begin=time.monotonic()
env=dict(os.environ,PYTHONPATH={OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
results=[]
for item in {specs!r}:
 run=root/item['run'];log=run/'scratch'/{(label+'.log')!r};log.parent.mkdir(exist_ok=True)
 args=[{OLD['python']!r},str(root/item['worker']),'--run',item['run'],'--output',item['run']+'/artifacts/{label}/worker']
 if {mechanics!r}:args+=['--mechanics']
 with log.open('xb') as stream:
  p=subprocess.run(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
 results.append(dict(run=item['run'],returncode=p.returncode,elapsed_seconds=time.monotonic()-begin))
 status=root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch'/{('h049-'+label+'-status.json')!r}
 status.parent.mkdir(exist_ok=True);status.write_text(json.dumps(dict(results=results,elapsed_seconds=time.monotonic()-begin)))
 if p.returncode:raise SystemExit(p.returncode)
'''
    result=remote(f'''from pathlib import Path
import json,subprocess,datetime
root=Path({REMOTE!r});scratch=root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch';scratch.mkdir(parents=True,exist_ok=True)
driver=scratch/{('h049-'+label+'-driver.py')!r}
assert not driver.exists();driver.write_text({driver!r})
with (scratch/{('h049-'+label+'.log')!r}).open('xb') as log:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',{str(600 if mechanics else 5400)!r},'python3',str(driver)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,phase={label!r},root=str(root),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds={600 if mechanics else 5400})))''')
    dump(receipt,result);print(json.dumps(result))

def status():
    result=remote(f'''from pathlib import Path
import json,subprocess
root=Path({REMOTE!r});results=[]
for rel in {[r.relative_to(ROOT).as_posix() for r in RUNS]!r}:
 run=root/rel
 for phase in ['mechanics-01','attempt-01']:
  worker=run/'artifacts'/phase/'worker'
  if not worker.exists():continue
  row=dict(run=rel,phase=phase)
  for name in ['progress.json','manifest.json']:
   if (worker/name).exists():row[name]=json.loads((worker/name).read_text())
  if 'progress.json' in row:
   arm=row['progress.json'].get('arm')
   if arm and (worker/arm/'progress.json').exists():row['arm_progress']=json.loads((worker/arm/'progress.json').read_text())
  if 'manifest.json' in row:
   m=row['manifest.json'];row['manifest.json']=dict(status=m['status'],completed_arms=len(m['results']),elapsed_seconds=m.get('elapsed_seconds'),failed=[r['arm']['id'] for r in m['results'] if r['status']!='completed'])
  log=run/'scratch'/(phase+'.log')
  if log.exists():row['log_tail']=log.read_text()[-700:]
  results.append(row)
print(json.dumps(dict(workers=results,gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)))''')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['stage','smoke','launch','status']);q=a.parse_args()
    if q.action=='stage':stage()
    elif q.action=='status':status()
    else:launch(q.action=='smoke')
