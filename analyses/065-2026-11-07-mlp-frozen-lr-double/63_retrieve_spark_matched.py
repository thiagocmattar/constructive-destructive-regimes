"""Inventory durable raw evidence on Spark; fetch verified compact artifacts."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fleet',HERE/'58_matched_fleet.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def retrieve(case):
    run=f.RUNS[case];rel=run+'/artifacts/attempt-01'
    script=f'''set -e
cd {f.SPARKROOT}
{f.SPARKPY} - <<'PY'
import json,hashlib,tarfile
from pathlib import Path
root=Path('.');out=root/{rel!r};files=[]
for p in sorted(out.rglob('*')):
 if p.is_file() and p.name!='durable_inventory.json':
  with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
  files.append(dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=digest))
inventory=out/'durable_inventory.json';inventory.write_text(json.dumps(dict(files=files),indent=2))
archive=Path('scratch/{case}-complete-compact.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.suffix in ['.json','.jsonl','.npz']:t.add(p,arcname=p.as_posix())
with archive.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
print(json.dumps(dict(status='inventoried',root=str(out.resolve()),files=len(files),raw_bytes=sum(v['bytes'] for v in files),archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=digest)))
PY
'''
    receipt=json.loads(f.spark(script,timeout=600).strip().splitlines()[-1])
    local=f.S/(case+'-complete-compact.tar.gz')
    result=subprocess.run(['scp','-q','-o','BatchMode=yes',f.SPARK+':'+f.SPARKROOT+'/'+receipt['archive'],str(local)],capture_output=True,timeout=600,creationflags=f.x.NO_WINDOW)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    assert local.stat().st_size==receipt['archive_bytes'] and f.x.file_hash(local)==receipt['archive_sha256']
    with tarfile.open(local) as t:t.extractall(f.ROOT,filter='data')
    inventory=json.loads((f.ROOT/rel/'durable_inventory.json').read_text())
    verified=0
    for item in inventory['files']:
        path=f.ROOT/rel/item['path']
        if path.suffix in ['.json','.jsonl','.npz']:
            assert path.stat().st_size==item['bytes'] and f.x.file_hash(path)==item['sha256']
            verified+=1
    receipt.update(status='passed',compact_files_verified=verified,retained_raw_location=f.SPARK+':'+f.SPARKROOT+'/'+rel,utc=f.x.stamp())
    f.dump(f.ROOT/run/'spark_durable_retention.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=list(f.RUNS));retrieve(p.parse_args().case)
