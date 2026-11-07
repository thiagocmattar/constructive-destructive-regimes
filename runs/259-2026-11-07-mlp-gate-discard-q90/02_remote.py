"""Stage, detach, inspect and retrieve only Run259 on the owned Spark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOST = 'tcml-spark@spark-2277.local'
PREVIOUS = json.loads((ROOT/'runs/228-2026-11-04-mlp-global-clipping/spark_launch.json').read_text())
REMOTE = PREVIOUS['root']
PYTHON = PREVIOUS['python']
PYTHONPATH = PREVIOUS['pythonpath']
REL = HERE.relative_to(ROOT).as_posix()


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def remote(code, timeout=60):
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST,'python3','-'],
                       input=code, text=True, capture_output=True, timeout=timeout, check=True)
    return json.loads(r.stdout)


def stage():
    assert not (HERE/'spark_launch.json').exists(), 'Do not relaunch an executed run'
    files=[]
    previous=ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear'
    for item in json.loads((previous/'input_manifest.json').read_text())['files']:
        p=ROOT/item.get('local_path',item['path'])
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
        files.append(item)
    for name in ['fork.pt','boundaries.jsonl','preactivation1.npy','hidden1.npy','preactivation2.npy','hidden2.npy','output.npy','raw_gradient.npy']:
        p=previous/'artifacts/spark/payload/worker'/name
        files.append(dict(path=(previous.relative_to(ROOT)/'artifacts/attempt-01/worker'/name).as_posix(),
            local_path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ['01_worker.py','config.json','test_intervention.py']:
        p=HERE/name;files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ['frozen_first_linear.py','selective_first_linear.py','selective_continuation.py']:
        p=ROOT/'src/eos_tasks'/name
        files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    cfg=json.loads((HERE/'config.json').read_text())
    calibration=ROOT/cfg['calibration_run']
    p=calibration/'config.json';files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    for folder,names in [(calibration,['thresholds.json','manifest.json'])]+([(ROOT/cfg['target_run'],['updates.jsonl','manifest.json'])] if cfg['target_run'] else []):
        for name in names:
            p=folder/'artifacts/spark/payload/worker'/name
            files.append(dict(path=(folder.relative_to(ROOT)/'artifacts/attempt-01/worker'/name).as_posix(),local_path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    dump(HERE/'input_manifest.json',dict(run_id='259',files=files))
    scratch = HERE/'scratch'
    scratch.mkdir(exist_ok=True)
    archive = scratch/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name in ['01_worker.py','config.json','test_intervention.py','input_manifest.json','README.md']:
            t.add(HERE/name,arcname=REL+'/'+name)
        for name in ['frozen_first_linear.py','selective_first_linear.py','selective_continuation.py']:
            t.add(ROOT/'src/eos_tasks'/name,arcname='src/eos_tasks/'+name)
    target = REMOTE+'/run259-source.tar.gz'
    subprocess.run(['scp','-q',str(archive),HOST+':'+target],check=True,timeout=60)
    result = remote(f'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
root=Path({REMOTE!r}); run=root/{REL!r}; archive=Path({target!r})
assert not run.exists(), 'Run259 destination already exists'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={sha(archive)!r}
shared=root/'src/eos_tasks/frozen_first_linear.py'
assert shared.exists() and hashlib.sha256(shared.read_bytes()).hexdigest()=={sha(ROOT/'src/eos_tasks/frozen_first_linear.py')!r}
continuation=root/'src/eos_tasks/selective_continuation.py'
assert not continuation.exists() or hashlib.sha256(continuation.read_bytes()).hexdigest()=={sha(ROOT/'src/eos_tasks/selective_continuation.py')!r}
with tarfile.open(archive) as t:
 assert all(x.name.startswith({(REL+'/')!r}) or x.name in ['src/eos_tasks/frozen_first_linear.py','src/eos_tasks/selective_first_linear.py','src/eos_tasks/selective_continuation.py'] for x in t.getmembers())
 t.extractall(root,filter='data')
checked=0
for item in json.loads((run/'input_manifest.json').read_text())['files']:
 p=root/item['path']
 assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'], str(p)
 checked+=1
env=dict(os.environ,PYTHONPATH={PYTHONPATH!r},PYTHONUNBUFFERED='1')
r=subprocess.run([{PYTHON!r},str(run/'test_intervention.py')],cwd=root,env=env,capture_output=True,text=True,timeout=45)
assert r.returncode==0, r.stderr
print(json.dumps(dict(status='passed',input_files_checked=checked,tests=json.loads(r.stdout),
 source_archive_sha256={sha(archive)!r},root=str(root),python={PYTHON!r})))
''')
    dump(HERE/'verification_prelaunch.json',result)
    print(json.dumps(result))


def launch():
    assert json.loads((HERE/'verification_prelaunch.json').read_text())['status']=='passed'
    assert not (HERE/'spark_launch.json').exists()
    result = remote(f'''
from pathlib import Path
import datetime,json,os,subprocess
root=Path({REMOTE!r}); run=root/{REL!r}
assert not (run/'artifacts').exists(), 'Execution already exists'
logs=run/'scratch';logs.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH={PYTHONPATH!r},PYTHONUNBUFFERED='1')
with (logs/'worker.log').open('xb') as log:
 p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30','630',{PYTHON!r},str(run/'01_worker.py')],
  cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(root),python={PYTHON!r},run={REL!r},
 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds=630)))
''')
    dump(HERE/'spark_launch.json',result)
    print(json.dumps(result))


def status():
    print(json.dumps(remote(f'''
from pathlib import Path
import json,subprocess
run=Path({REMOTE!r})/{REL!r}; out=run/'artifacts/attempt-01/worker';p=out/'manifest.json'
record=json.loads(p.read_text()) if p.exists() else dict(status='starting')
record['progress']=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
record['log_tail']=(run/'scratch/worker.log').read_text()[-3500:]
print(json.dumps(record))
'''),indent=2))


def collect():
    receipt = remote(f'''
from pathlib import Path
import json,hashlib,tarfile
root=Path({REMOTE!r});run=root/{REL!r};p=run/'artifacts/attempt-01'
m=json.loads((p/'worker/manifest.json').read_text());assert m['status']!='running'
files=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='transfer_inventory.json':
  files.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(p/'transfer_inventory.json').write_text(json.dumps(dict(files=files)))
archive=run/'scratch/output.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 t.add(p,arcname='payload');t.add(run/'scratch/worker.log',arcname='worker.log')
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),status=m['status'])))
''', timeout=600)
    target=HERE/'artifacts/spark'
    target.mkdir(parents=True,exist_ok=True)
    archive=target/'output.tar.gz'
    subprocess.run(['scp','-q',HOST+':'+receipt['path'],str(archive)],check=True,timeout=600)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    physical=Path('\\\\?\\'+str(target.resolve())) if os.name=='nt' else target
    with tarfile.open(archive) as t:
        t.extractall(physical,filter='data')
    files=json.loads((target/'payload/transfer_inventory.json').read_text())['files']
    for f in files:
        p=target/'payload'/f['path']
        assert p.resolve().is_relative_to((target/'payload').resolve())
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
    receipt.update(verified=True,files=len(files),file_bytes=sum(f['bytes'] for f in files))
    dump(HERE/'retrieval.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['stage','launch','status','collect'])
    globals()[parser.parse_args().action]()
