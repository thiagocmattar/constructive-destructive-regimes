"""Snapshot/verify task-owned Spark artifacts; use --final after the worker finishes."""
import argparse,hashlib,json,os,subprocess,tarfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
HOST='tcml-spark@spark-2277.local'
REMOTE='/home/tcml-spark/experiments/constructive-destructive-regimes/interventions-2026-11-03'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');mode='final' if args.final else 'scalars'
 remote=r"""from pathlib import Path
import hashlib,io,json,tarfile,time
root=Path('/home/tcml-spark/experiments/constructive-destructive-regimes/interventions-2026-11-03')
run=root/'runs/222-2026-11-03-spark-matched-interventions';out=run/'artifacts/attempts/001-spark'
final=FINAL
manifest=json.loads((out/'manifest.json').read_text())
if final:assert manifest['status']!='running' and (out/'inventory.json').exists(),'Terminal inventory not ready.'
files=[p for p in sorted((run/'artifacts').rglob('*')) if p.is_file() and not p.name.endswith('.tmp') and (final or p.suffix in ('.json','.jsonl'))]
for rel in ['run222_launch.json','profile_launch.json','profile_retry.json','profile.log','profile2.log','warm_qualification.log','run222.log','runs/221-2026-11-03-spark-fixed-step-interventions/artifacts/profile.json','runs/221-2026-11-03-spark-fixed-step-interventions/artifacts/warm_parent_qualification.json']:
 p=root/rel
 if p.is_file():files.append(p)
archive=root/'run222-transfer-MODE.tar.gz';inventory=[]
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p in files:
  assert p.resolve().is_relative_to(root.resolve())
  rel=p.relative_to(root).as_posix()
  if final:
   with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   size=p.stat().st_size;tar.add(p,arcname=rel,recursive=False)
  else:
   data=p.read_bytes()
   if p.suffix=='.jsonl' and data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
   digest=hashlib.sha256(data).hexdigest();size=len(data);info=tarfile.TarInfo(rel);info.size=size;tar.addfile(info,io.BytesIO(data))
  inventory.append(dict(path=rel,bytes=size,sha256=digest))
 data=json.dumps(dict(final=final,status=manifest['status'],files=inventory),indent=2).encode();info=tarfile.TarInfo('transfer_inventory.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest,status=manifest['status'],files=len(inventory),final=final)))
""".replace('FINAL',repr(args.final)).replace('MODE',mode)
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3','-'],input=remote,text=True,capture_output=True,check=True)
 receipt=json.loads(result.stdout);folder=HERE/'artifacts/snapshots'/stamp;folder.mkdir(parents=True,exist_ok=False)
 archive=folder/'transfer.tar.gz';subprocess.run(['scp','-q',HOST+':'+receipt['path'],str(archive)],check=True)
 assert sha(archive)==receipt['sha256']
 payload=folder/'payload';payload.mkdir()
 physical=Path('\\\\?\\'+str(payload.resolve())) if os.name=='nt' else payload
 with tarfile.open(archive) as tar:tar.extractall(physical,filter='data')
 inventory=json.loads((physical/'transfer_inventory.json').read_text())
 for item in inventory['files']:
  p=physical/item['path'];assert p.resolve().is_relative_to(physical.resolve())
  assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
 receipt.update(verified=True,retrieved_utc=stamp,payload_root=payload.relative_to(ROOT).as_posix(),retained_bytes=sum(x['bytes'] for x in inventory['files']))
 (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (HERE/'artifacts/latest_snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (HERE/'retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (HERE/'status.json').write_text(json.dumps(dict(status=receipt['status'],source='verified Spark snapshot',retrieved_utc=stamp,final=receipt['final']),indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
