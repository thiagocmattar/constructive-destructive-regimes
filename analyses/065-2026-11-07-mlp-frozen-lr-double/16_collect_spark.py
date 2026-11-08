"""Retrieve completed H048 Spark evidence with archive and per-file identities."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transfer',HERE/'14_prepare_transfer.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def collect():
    run=x.RUNS['mlp'];rel=run.relative_to(x.ROOT).as_posix()
    receipt=x.remote(f'''import json,hashlib,tarfile
from pathlib import Path
run=Path({x.REMOTE!r})/{rel!r};root=run/'artifacts'
assert json.loads((root/'scientific/worker/manifest.json').read_text())['status']=='completed'
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p!=root/'transfer_inventory.json':
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h))
inventory=root/'transfer_inventory.json';inventory.write_text(json.dumps(dict(files=files)))
archive=run/'scratch/results.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for f in files:t.add(root/f['path'],arcname=f['path'])
 t.add(inventory,arcname='transfer_inventory.json')
with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h,files=len(files),raw_bytes=sum(f['bytes'] for f in files))))''',timeout=600)
    target=run/'artifacts/spark';target.mkdir(parents=True,exist_ok=True)
    archive=run/'artifacts/spark-results.tar.gz'
    subprocess.run(['scp','-q',x.HOST+':'+receipt['archive'],str(archive)],check=True,timeout=900)
    assert archive.stat().st_size==receipt['bytes'] and x.sha(archive)==receipt['sha256']
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    for f in json.loads((target/'transfer_inventory.json').read_text())['files']:
        p=target/f['path'];assert p.stat().st_size==f['bytes'] and x.sha(p)==f['sha256'],str(p)
    assert archive.resolve().is_relative_to((run/'artifacts').resolve())
    archive.unlink()
    receipt.update(verified=True,local=target.relative_to(x.ROOT).as_posix(),duplicate_local_archive_removed=True)
    x.dump(run/'spark_retrieval.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':collect()
