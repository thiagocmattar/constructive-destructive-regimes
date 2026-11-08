"""Freeze case-specific inputs and stage/launch the MLP study on owned Spark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'runs/223-2026-11-04-multispike-interventions'
RUNS={c:ROOT/f'runs/{n}-2026-11-07-{c}-representation-interventions' for n,c in [(269,'mlp'),(270,'resnet'),(271,'transformer')]}
OLD=json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
HOST='tcml-spark@spark-2277.local';REMOTE=OLD['root']


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n',newline='\n')


def manifest(case):
    run=RUNS[case];files={}
    def add(p,target=None):
        target=target or p.relative_to(ROOT).as_posix()
        item=dict(path=target,bytes=p.stat().st_size,sha256=sha(p))
        if target!=p.relative_to(ROOT).as_posix():item['local_path']=p.relative_to(ROOT).as_posix()
        files[target]=item
    for f in json.loads((PARENT/'input_manifest.json').read_text())['files']:
        if case in f['tasks']:
            p=ROOT/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];add(p)
    for n in ['config.json','input_manifest.json','01_worker.py']:add(PARENT/n)
    for n in ['representation_policy.py','representation_continuation.py']:add(ROOT/'src/eos_tasks'/n)
    add(RUNS['mlp']/'test_policy.py');add(run/'config.json')
    for extra in json.loads((run/'config.json').read_text()).get('extra_inputs',[]):add(run/extra)
    for shared in json.loads((run/'config.json').read_text()).get('shared_inputs',[]):add(ROOT/shared)
    if case=='mlp':
        cfg=json.loads((run/'config.json').read_text())
        if cfg['fork']==448 and cfg['seed']==0:add(ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/spark/payload/worker/fork.pt',run.relative_to(ROOT).as_posix()+'/inputs/reference448.pt')
    else:
        worker=json.loads((run/'config.json').read_text())['parent_workers'][case]
        raw=PARENT/'artifacts'/worker/'payload/worker'
        add(raw/'prefix/end.pt',run.relative_to(ROOT).as_posix()+'/inputs/prefix.pt')
        add(raw/'control/boundaries.jsonl',run.relative_to(ROOT).as_posix()+'/inputs/boundaries.jsonl')
    result=dict(case=case,files=list(files.values()))
    if (run/'input_manifest.json').exists():assert json.loads((run/'input_manifest.json').read_text())==result,'Executed inputs may not be replaced'
    else:dump(run/'input_manifest.json',result)
    return result


def packet(case):
    run=RUNS[case];m=manifest(case);scratch=run/'scratch';scratch.mkdir(exist_ok=True);target=scratch/'inputs.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=1) as tar:
        for f in m['files']:tar.add(ROOT/f.get('local_path',f['path']),arcname=f['path'])
        tar.add(run/'input_manifest.json',arcname=run.relative_to(ROOT).as_posix()+'/input_manifest.json')
    receipt=dict(path=target.relative_to(ROOT).as_posix(),bytes=target.stat().st_size,sha256=sha(target))
    dump(run/'packet.json',receipt);print(case,receipt,flush=True)


def remote(code,timeout=90):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,'python3','-'],input=code,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
    return json.loads(p.stdout)


def spark_stage(case='mlp'):
    run=RUNS[case];m=manifest(case);rel=run.relative_to(ROOT).as_posix()
    needed=remote(f'''import json,hashlib
from pathlib import Path
root=Path({REMOTE!r});files={m['files']!r};missing=[]
for f in files:
 p=root/f['path']
 if not p.exists():missing.append(f['path'])
 else:
  assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],'Existing input differs: '+str(p)
print(json.dumps(missing))''')
    scratch=run/'scratch';scratch.mkdir(exist_ok=True);archive=scratch/'spark-inputs.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for f in m['files']:
            if f['path'] in needed:tar.add(ROOT/f.get('local_path',f['path']),arcname=f['path'])
        tar.add(run/'input_manifest.json',arcname=rel+'/input_manifest.json')
    dest=REMOTE+'/run269-inputs.tar.gz';subprocess.run(['scp','-q',str(archive),HOST+':'+dest],check=True,timeout=240)
    result=remote(f'''import json,hashlib,tarfile,os,subprocess
from pathlib import Path
root=Path({REMOTE!r});archive=Path({dest!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with tarfile.open(archive) as t:t.extractall(root,filter='data')
for f in {m['files']!r}:
 p=root/f['path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
env=dict(os.environ,PYTHONPATH={OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8')
p=subprocess.run([{OLD['python']!r},str(root/{RUNS['mlp'].relative_to(ROOT).as_posix()!r}/'test_policy.py')],cwd=root,env=env,capture_output=True,text=True)
assert p.returncode==0,p.stderr
print(json.dumps(dict(status='passed',input_files={len(m['files'])},uploaded={len(needed)},tests=json.loads(p.stdout))))''',timeout=180)
    dump(run/'spark_preflight.json',result);print(result)


def spark_launch(mechanics,case='mlp',arms=None):
    run=RUNS[case];rel=run.relative_to(ROOT).as_posix();label='mechanics' if mechanics else 'scientific'
    worker=json.loads((run/'config.json').read_text()).get('worker_script')
    entry=[rel+'/'+worker] if worker else ['-m','src.eos_tasks.representation_continuation']
    if mechanics:
        retry=0
        while (run/(label+'_launch.json')).exists():
            retry+=1;label=f'mechanics-retry{retry:02d}'
    receipt=run/(label+'_launch.json');assert not receipt.exists()
    result=remote(f'''import json,os,subprocess,datetime
from pathlib import Path
root=Path({REMOTE!r});run=root/{rel!r};(run/'scratch').mkdir(exist_ok=True)
args=[{OLD['python']!r}]+{entry!r}+['--run',{rel!r},'--output',{(rel+'/artifacts/'+label+'/worker')!r}]
if {mechanics!r}:args+=['--mechanics','--arms',{arms or 'control,functional_entry,block_shock'!r}]
env=dict(os.environ,PYTHONPATH={OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (run/'scratch'/{(label+'.log')!r}).open('xb') as log:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',{str(5400 if case=='mlp' else 21600)!r}]+args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,phase={label!r},root=str(root),python={OLD['python']!r},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds={5400 if case=='mlp' else 21600})))''')
    dump(receipt,result);print(result)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['packet','stage','launch','mechanics']);a.add_argument('--case',choices=list(RUNS));q=a.parse_args()
    if q.action=='packet':packet(q.case)
    elif q.action=='stage':spark_stage(q.case or 'mlp')
    else:spark_launch(q.action=='mechanics',q.case or 'mlp')
