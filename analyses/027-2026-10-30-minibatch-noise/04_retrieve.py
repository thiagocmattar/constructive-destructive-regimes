"""Package completed task evidence, retrieve and verify before cloud cleanup."""
from pathlib import Path, PurePosixPath
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).parent;ROOT=HERE.resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.artifact_inventory import verify
spec=importlib.util.spec_from_file_location('control',HERE/'01_control.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
RUN185='runs/185-2026-10-30-minibatch-noise-calibration/artifacts/attempt-01'
RUN186='runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01'
CACHE=ROOT/'cache/minibatch-noise-preparation'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def download_parts(receipt,archive):
    """Four independent SSH byte ranges; the archive hash checks their assembly."""
    assert receipt['path']=='/workspace/h023-evidence.tar.gz'
    folder=CACHE/'h023-transfer-parts';folder.mkdir(exist_ok=True)
    block=(receipt['bytes']+3)//4;l=control.lease()
    def fetch(index):
        begin=index*block;count=min(block,receipt['bytes']-begin)
        target=folder/f'part{index:02d}'
        if not target.exists() or target.stat().st_size!=count:
            script=f"""import sys
with open('/workspace/h023-evidence.tar.gz','rb') as source:
 source.seek({begin})
 remaining={count}
 while remaining:
  chunk=source.read(min(remaining,1048576))
  if not chunk:raise RuntimeError('Unexpected archive EOF')
  sys.stdout.buffer.write(chunk)
  remaining-=len(chunk)
"""
            with target.open('wb') as stream:
                result=subprocess.run(control.transport()+[f"root@{l['host']}",'python3 -c '+shlex.quote(script)],
                                      stdout=stream,stderr=subprocess.PIPE,timeout=5400)
            assert result.returncode==0,result.stderr.decode(errors='replace')[-2000:]
        assert target.stat().st_size==count
        return target
    with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(fetch,range(4)))
    with archive.open('wb') as target:
        for part in parts:
            with part.open('rb') as source:shutil.copyfileobj(source,target,1048576)
    assert archive.stat().st_size==receipt['bytes'] and digest(archive)==receipt['sha256']
    for part in parts:
        assert part.resolve().is_relative_to(CACHE.resolve())
        part.unlink()


def prepare():
    state=control.status()
    assert state.get('manifest',{}).get('status')=='complete',state
    assert state.get('production_exit')=='0',state
    # All paths are literal task-owned paths. The running experiment is finished.
    script=f'''cd {control.REMOTE}
cat > worker-control/package.py <<'PY'
import hashlib,json,shutil,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.artifact_inventory import create
root=Path.cwd();attempt=root/'{RUN186}'
assert json.loads((attempt/'production/manifest.json').read_text())['status']=='complete'
metadata=attempt/'execution';metadata.mkdir(exist_ok=True)
for name in ('setup.sh','setup.log','setup.exit','calibration.sh','calibration.log','calibration.exit','production.sh','production.log','production.exit','environment-lock.txt'):
 shutil.copy2(root/'worker-control'/name,metadata/name)
source=metadata/'source';source.mkdir(exist_ok=True)
inputs=json.loads((root/'runs/186-2026-10-30-mlp-minibatch-noise/input_manifest.json').read_text())
for name,entry in inputs.items():
 p=root/name
 assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
 if name.endswith('.py') or name.endswith('/config.json'):
  target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
shutil.copy2(root/'runs/186-2026-10-30-mlp-minibatch-noise/input_manifest.json',metadata/'input_manifest.json')
for relative in ('{RUN185}','{RUN186}/production','{RUN186}/execution'):
 folder=root/relative;create(folder,folder/'inventory.json')
archive=Path('/workspace/h023-evidence.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for relative in ('{RUN185}','{RUN186}'):
  tar.add(root/relative,arcname=relative)
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha,created_epoch=time.time())
(root/'worker-control/package.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt),flush=True)
PY
nohup /opt/h023-env/bin/python -u worker-control/package.py > worker-control/package.log 2>&1 < /dev/null &
echo $!
'''
    print(control.ssh(script),flush=True)


def retrieve():
    CACHE.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(control.ssh('cat '+control.REMOTE+'/worker-control/package.json\n'))
    control.save(HERE/'remote_archive.json',receipt)
    archive=CACHE/'h023-evidence.tar.gz'
    if not archive.exists() or archive.stat().st_size!=receipt['bytes'] or digest(archive)!=receipt['sha256']:
        download_parts(receipt,archive)
    # Existing complete archives or freshly assembled ranges were hashed above.
    assert archive.stat().st_size==receipt['bytes']
    prefixes=(RUN185+'/',RUN186+'/')
    # Validate every entry before extracting, and never allow links or path escapes.
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            name=member.name;p=PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,name
            assert name in (RUN185,RUN186) or name.startswith(prefixes),name
            assert member.isfile() or member.isdir(),name
            target=ROOT.joinpath(*p.parts).resolve();assert target.is_relative_to(ROOT),name
        tar.extractall(ROOT,filter='data')
    receipts={}
    for relative in (RUN185,RUN186+'/production',RUN186+'/execution'):
        folder=ROOT/relative;result=verify(folder,folder/'inventory.json',strict=True)
        assert result['verified'],result;receipts[relative]=result
    record=dict(verified=True,verified_epoch=time.time(),archive=receipt,inventories=receipts,
                total_verified_bytes=sum(v['total_bytes'] for v in receipts.values()))
    control.save(HERE/'retrieval.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','retrieve']);a=p.parse_args()
    prepare() if a.action=='prepare' else retrieve()
