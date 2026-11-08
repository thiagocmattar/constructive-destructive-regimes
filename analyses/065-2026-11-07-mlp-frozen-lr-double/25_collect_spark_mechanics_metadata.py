"""Keep metadata locally for the slow Spark Transformer mechanics; raw stays on Spark."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE/'14_prepare_transfer.py');x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/274-2026-11-07-transformer-distributed-displacement');args=parser.parse_args();RUN=args.run
    result=x.remote(f'''import hashlib,json,tarfile
from pathlib import Path
run=Path({x.REMOTE!r})/{RUN!r};root=run/'artifacts/mechanics/worker'
assert json.loads((root/'manifest.json').read_text())['status']=='completed'
raw=[];metadata=[]
for p in sorted(root.rglob('*')):
 if p.is_file():
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  item=dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=h);raw.append(item)
  if p.suffix in ['.json','.jsonl']:metadata.append(item)
inventory=run/'scratch/mechanics-raw-inventory.json';inventory.write_text(json.dumps(dict(root=str(root),files=raw)))
archive=run/'scratch/mechanics-metadata.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for item in metadata:t.add(root/item['path'],arcname=item['path'])
 t.add(inventory,arcname='retained_raw_inventory.json');t.add(run/'scratch/mechanics.log',arcname='infrastructure/mechanics.log')
with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=h,metadata=metadata,raw_root=str(root),raw_bytes=sum(r['bytes'] for r in raw),raw_files=len(raw))))''',timeout=300)
    target=x.ROOT/RUN/'artifacts/spark-mechanics-metadata';target.mkdir(exist_ok=True,parents=True);archive=target/'metadata.tar.gz'
    subprocess.run(['scp','-q',x.HOST+':'+result['archive'],str(archive)],check=True,timeout=120)
    assert archive.stat().st_size==result['bytes'] and x.sha(archive)==result['sha256']
    with tarfile.open(archive) as t:t.extractall(target,filter='data')
    for item in result['metadata']:
        p=target/item['path'];assert p.stat().st_size==item['bytes'] and x.sha(p)==item['sha256']
    archive.unlink();result.update(metadata_verified=True,raw_downloaded=False,raw_retained_on='owned Spark',local=target.relative_to(x.ROOT).as_posix())
    x.dump(x.ROOT/RUN/'spark_mechanics_metadata_retrieval.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='metadata'}))


if __name__=='__main__':main()
