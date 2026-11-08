"""Stage follow-up inputs by verified identity, reusing immutable on-pod caches."""
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'15_cloud_transfer.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    pod=c.get(args.name);run=args.run.resolve();rel=run.relative_to(c.ROOT).as_posix();manifest=json.loads((run/'input_manifest.json').read_text())
    code=f'''python3 - <<'PY'
import json,hashlib,shutil
from pathlib import Path
root=Path({c.REMOTE!r});known={{}};missing=[];reused=[]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for m in (root/'runs').glob('*/input_manifest.json'):
 for f in json.loads(m.read_text())['files']:
  p=root/f['path']
  if p.exists():known[f['sha256']]=p
for f in {manifest['files']!r}:
 p=root/f['path']
 if p.exists():assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
 elif f['sha256'] in known:
  source=known[f['sha256']];assert sha(source)==f['sha256'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,p);reused.append(f['path'])
 else:missing.append(f['path'])
print(json.dumps(dict(missing=missing,reused=reused)))
PY
'''
    need=json.loads(c.x.ssh(pod,code,timeout=180));archive=run/'scratch'/('delta-'+args.name+'.tar.gz');archive.parent.mkdir(exist_ok=True)
    observer=HERE/'18_verify_representations.py'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for f in manifest['files']:
            if f['path'] in need['missing']:tar.add(c.ROOT/f.get('local_path',f['path']),arcname=f['path'])
        tar.add(run/'input_manifest.json',arcname=rel+'/input_manifest.json')
        tar.add(observer,arcname=observer.relative_to(c.ROOT).as_posix())
    digest=c.x.file_hash(archive);dest='/workspace/'+archive.name;c.x.copy(pod,archive,dest)
    result=c.x.ssh(pod,f'''python3 - <<'PY'
import json,hashlib,tarfile
from pathlib import Path
root=Path({c.REMOTE!r});archive=Path({dest!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
with tarfile.open(archive) as t:t.extractall(root,filter='data')
for f in {manifest['files']!r}:
 p=root/f['path']
 with p.open('rb') as h:actual=hashlib.file_digest(h,'sha256').hexdigest()
 assert p.stat().st_size==f['bytes'] and actual==f['sha256'],str(p)
print(json.dumps(dict(status='passed',files={len(manifest['files'])})))
PY
''',timeout=180)
    receipt=dict(**json.loads(result),delta_bytes=archive.stat().st_size,delta_sha256=digest,**need)
    c.dump(run/(args.name+'_preflight.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
