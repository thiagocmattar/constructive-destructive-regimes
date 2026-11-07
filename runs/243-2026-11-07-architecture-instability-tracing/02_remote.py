"""Task-scoped staging and detached execution on the owned Spark."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; REL=HERE.relative_to(ROOT).as_posix()
HOST='tcml-spark@spark-2277.local'
OLD=json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
REMOTE=OLD['root']; PYTHON=OLD['python']; PP=OLD['pythonpath']

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',newline='\n')
def remote(code,timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,'python3','-'],input=code,text=True,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
    return json.loads(r.stdout)

def stage():
    parent=ROOT/'runs/223-2026-11-04-multispike-interventions'; cfg=json.loads((HERE/'config.json').read_text()); files={}
    for f in json.loads((parent/'input_manifest.json').read_text())['files']:
        if set(f['tasks']) & {'resnet','transformer'}:
            p=ROOT/f['path']; assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
            files[f['path']]=dict(path=f['path'],bytes=p.stat().st_size,sha256=sha(p),local=str(p))
    for name in ['01_worker.py','input_manifest.json','config.json']:
        p=parent/name; files[p.relative_to(ROOT).as_posix()]=dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),local=str(p))
    for case,sp in cfg['cases'].items():
        raw=parent/'artifacts'/sp['parent_worker']/'payload/worker'
        for name,source in [('prefix.pt','prefix/end.pt'),('boundaries.jsonl','control/boundaries.jsonl')]:
            p=raw/source; target=REL+'/inputs/'+case+'/'+name
            files[target]=dict(path=target,bytes=p.stat().st_size,sha256=sha(p),local=str(p))
    for name in ['01_worker.py','02_remote.py','diagnostics.py','config.json']:
        p=HERE/name; target=p.relative_to(ROOT).as_posix();files[target]=dict(path=target,bytes=p.stat().st_size,sha256=sha(p),local=str(p))
    manifest=dict(files=[{k:v for k,v in f.items() if k!='local'} for f in files.values()])
    dump(HERE/'input_manifest.json',manifest)
    needed=remote(f'''
from pathlib import Path
import json,hashlib
root=Path({REMOTE!r}); required={manifest['files']!r}; missing=[]
for f in required:
 p=root/f['path']
 if not p.exists() or p.stat().st_size!=f['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:
  assert f['path'].startswith({REL!r}) or not p.exists(), 'Existing shared input differs: '+str(p)
  missing.append(f['path'])
print(json.dumps(missing))
''')
    (HERE/'scratch').mkdir(exist_ok=True); archive=HERE/'scratch/source.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for path in needed:t.add(files[path]['local'],arcname=path)
        t.add(HERE/'input_manifest.json',arcname=REL+'/input_manifest.json')
    target=REMOTE+'/run243-source.tar.gz';subprocess.run(['scp','-q',str(archive),HOST+':'+target],check=True,timeout=180)
    print(remote(f'''
import json,tarfile,hashlib
from pathlib import Path
root=Path({REMOTE!r}); archive=Path({target!r});assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
with tarfile.open(archive) as t:t.extractall(root,filter='data')
for f in {manifest['files']!r}:
 p=root/f['path'];assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
print(json.dumps(dict(status='verified',files={len(files)},uploaded={len(needed)})))
''',timeout=120))

def launch(case,calibrate):
    label=case+('-calibration' if calibrate else '')
    receipt=HERE/(label+'_launch.json');assert not receipt.exists()
    r=remote(f'''
import json,os,subprocess,datetime
from pathlib import Path
root=Path({REMOTE!r});run=root/{REL!r};(run/'scratch').mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH={PP!r},PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
args=[{PYTHON!r},str(run/'01_worker.py'),'--case',{case!r}]+(['--calibrate'] if {calibrate!r} else [])
with (run/'scratch'/{(label+'.log')!r}).open('xb') as log:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30','5500']+args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,case={case!r},calibration={calibrate!r},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
''');dump(receipt,r);print(r)

def status():
    print(json.dumps(remote(f'''
import json
from pathlib import Path
run=Path({REMOTE!r})/{REL!r}
records={{str(p.relative_to(run)):json.loads(p.read_text()) for p in (run/'artifacts').glob('*/*/*.json') if p.name in ['manifest.json','progress.json','reduction_progress.json']}}
records['logs']={{p.name:p.read_text()[-1200:] for p in (run/'scratch').glob('*.log')}}
print(json.dumps(records))
'''),indent=2))

def collect():
    r=remote(f'''
import json,tarfile,hashlib
from pathlib import Path
run=Path({REMOTE!r})/{REL!r};root=run/'artifacts/attempt-01';files=[]
for case in ['resnet','transformer']:
 assert json.loads((root/case/'manifest.json').read_text())['status']=='completed'
for p in sorted(root.rglob('*')):
 if p.is_file():
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h,download=('raw' not in p.parts and 'checkpoints' not in p.parts and p.suffix!='.pt')))
(root/'inventory.json').write_text(json.dumps(dict(files=files)))
archive=run/'scratch/results.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 t.add(root/'inventory.json',arcname='inventory.json')
 for f in files:
  if f['download']:t.add(root/f['path'],arcname=f['path'])
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),retained_bytes=sum(f['bytes'] for f in files))))
''',timeout=600)
    target=HERE/'artifacts/spark';target.mkdir(parents=True,exist_ok=True);archive=target/'results.tar.gz'
    subprocess.run(['scp','-q',HOST+':'+r['archive'],str(archive)],check=True,timeout=120)
    assert archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256']
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    files=json.loads((target/'inventory.json').read_text())['files']
    for f in files:
        if f['download']:
            p=target/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    r['verified_downloads']=sum(f['download'] for f in files);dump(HERE/'retrieval.json',r);print(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','launch','status','collect']);p.add_argument('--case',choices=['resnet','transformer']);p.add_argument('--calibrate',action='store_true');a=p.parse_args()
    if a.action=='launch':launch(a.case,a.calibrate)
    else:globals()[a.action]()
