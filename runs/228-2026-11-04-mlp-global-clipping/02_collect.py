"""Collect terminal Spark Run228 evidence with archive and per-file SHA checks."""
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT_REMOTE='/home/tcml-spark/experiments/constructive-destructive-regimes/multispike-2026-11-04'
HOST='tcml-spark@spark-2277.local'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    rel=HERE.name
    code=f'''from pathlib import Path
import json,hashlib,tarfile
root=Path({ROOT_REMOTE!r});p=root/'runs'/{rel!r}/'artifacts'
m=json.loads((p/'worker/manifest.json').read_text());assert m['status']!='running'
rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name!='transfer_inventory.json':
  with f.open('rb') as s:h=hashlib.file_digest(s,'sha256').hexdigest()
  rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h))
(p/'transfer_inventory.json').write_text(json.dumps(dict(files=rows)))
archive=root/'run228-spark-output.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(p,arcname='payload')
with archive.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=str(archive),sha256=h,status=m['status'])))
'''
    r=subprocess.run(['ssh','-o','BatchMode=yes',HOST,'python3','-'],input=code.encode(),capture_output=True,check=True);remote=json.loads(r.stdout)
    target=HERE/'artifacts/spark';target.mkdir(exist_ok=True,parents=True);archive=target/'output.tar.gz'
    subprocess.run(['scp','-q',HOST+':'+remote['path'],str(archive)],check=True);assert sha(archive)==remote['sha256']
    physical=Path('\\\\?\\'+str(target.resolve())) if os.name=='nt' else target
    with tarfile.open(archive) as t:t.extractall(physical,filter='data')
    inventory=json.loads((physical/'payload/transfer_inventory.json').read_text())
    for f in inventory['files']:
        p=physical/'payload'/f['path'];assert p.resolve().is_relative_to((physical/'payload').resolve());assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    receipt=dict(**remote,verified=True,files=len(inventory['files']));(target/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
