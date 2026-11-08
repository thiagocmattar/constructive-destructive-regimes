"""Retrieve immutable completed branches; final job retrieval transfers only deltas."""
import importlib.util
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE/'15_cloud_transfer.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def collect(name,label,arm=None):
    pod=c.get(name);job=next(j for j in pod['jobs'] if j['label']==label);run_dir=job.get('run_dir',c.RUNS[job['case']])
    relative=job['output']+('/'+arm if arm else '');target=c.ROOT/relative
    receipt_path=c.ROOT/run_dir/(name+'_'+label+('_'+arm if arm else '')+'_retrieval.json')
    if arm and receipt_path.exists() and json.loads(receipt_path.read_text()).get('verified'):return
    log_names=[] if arm else [label+'.log',label+'.exit',label+'.sh',label+'-audit.log','setup.log','setup.exit']
    inventory_code=f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil
root=Path({c.REMOTE!r})/{relative!r};root.mkdir(parents=True,exist_ok=True)
if {arm=='shared'!r}:
 assert (root/'qualification.json').exists() and any(root.parent.glob('*/manifest.json'))
elif {arm is not None!r}:assert json.loads((root/'manifest.json').read_text())['status']=='completed'
if {arm=='control'!r} and json.loads((Path({c.REMOTE!r})/{run_dir!r}/'config.json').read_text()).get('baseline_reference'):
 assert (root.parent/'control_reuse.json').exists(),'Control copying/identity checks have not finished'
if {bool(log_names)!r}:
 logs=root/'infrastructure';logs.mkdir(exist_ok=True)
 for name in {log_names!r}:
  p=Path('/workspace')/name
  if p.exists():shutil.copy2(p,logs/name)
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p!=root/'transfer_inventory.json':
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h))
inventory=dict(files=files);(root/'transfer_inventory.json').write_text(json.dumps(inventory));print(json.dumps(inventory))
PY
'''
    inventory=json.loads(c.x.ssh(pod,inventory_code,timeout=300));missing=[]
    for f in inventory['files']:
        local=target/f['path']
        if not local.exists() or local.stat().st_size!=f['bytes'] or c.x.file_hash(local)!=f['sha256']:missing.append(f['path'])
    filename=name+'-'+label+('-'+arm if arm else '')+'-delta.tar.gz'
    archive_code=f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib,tarfile
root=Path({c.REMOTE!r})/{relative!r};archive=Path('/workspace')/{filename!r}
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for name in {missing!r}+['transfer_inventory.json']:t.add(root/name,arcname=name)
with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h)))
PY
'''
    receipt=json.loads(c.x.ssh(pod,archive_code,timeout=600));scratch=Path('R:/cdr-h048-transfers') if Path('R:/').is_dir() else target.parent
    scratch.mkdir(parents=True,exist_ok=True);archive=scratch/filename;c.x.copy(pod,archive,receipt['archive'],False)
    assert archive.stat().st_size==receipt['bytes'] and c.x.file_hash(archive)==receipt['sha256']
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    assert json.loads((target/'transfer_inventory.json').read_text())==inventory
    for f in inventory['files']:
        p=target/f['path'];assert p.stat().st_size==f['bytes'] and c.x.file_hash(p)==f['sha256'],str(p)
    assert archive.resolve().is_relative_to(scratch.resolve());archive.unlink()
    receipt.update(verified=True,local=relative,utc=c.x.stamp(),files=len(inventory['files']),raw_bytes=sum(f['bytes'] for f in inventory['files']),
        transferred_files=len(missing),duplicate_local_archive_removed=True,streamed_arm=arm)
    c.dump(receipt_path,receipt)
    if not arm:
        latest=c.get(name);next(j for j in latest['jobs'] if j['label']==label)['verified_retrieval']=True;c.update(latest)
    print(json.dumps(dict(name=name,label=label,arm=arm,retrieval=receipt)),flush=True)
