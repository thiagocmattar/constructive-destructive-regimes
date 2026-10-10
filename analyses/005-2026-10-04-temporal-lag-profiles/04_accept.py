"""Verify archive size/hash, every artifact and scientific identities before deletion."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def accept(run_name):
 run=ROOT/'runs'/run_name; pre=run/'prelaunch'; lease=json.loads((pre/'lease.json').read_text()); attempt=lease['attempt']
 archive=ROOT/'cache/runpod'/f'run{run_name[:3]}-{attempt}.tar.gz'
 expected_bytes=int(archive.with_name(archive.name+'.bytes').read_text())
 expected_sha=archive.with_name(archive.name+'.sha256').read_text().split()[0]
 assert archive.stat().st_size==expected_bytes and sha(archive)==expected_sha
 target=(run/'artifacts').resolve(); assert target.is_relative_to(ROOT.resolve())
 base=target/attempt
 with tarfile.open(archive,'r:gz') as incoming:
  members=incoming.getmembers(); seen=set()
  for member in members:
   p=PurePosixPath(member.name)
   assert p.parts and p.parts[0]==attempt and not p.is_absolute() and '..' not in p.parts
   assert '\\' not in member.name and ':' not in member.name and member.name not in seen
   assert member.isfile() or member.isdir(); seen.add(member.name)
  if base.exists():
   # Re-verification may follow a failed check; anchor the existing inventory
   # to this verified archive, and never overwrite already extracted evidence.
   archived_inventory=incoming.extractfile(attempt+'/inventory.json').read()
   assert (base/'inventory.json').read_bytes()==archived_inventory
  else:incoming.extractall(target,members=members,filter='data')
 sys.path.insert(0,str(ROOT/'tools'));from artifact_inventory import verify
 inventory=verify(base,base/'inventory.json',strict=True);assert inventory['verified'],inventory
 for p in (base/'source').glob('*'):
  if p.is_file():assert p.read_bytes()==(run/p.name).read_bytes(),p
 for p in (base/'source/baseline').glob('*.py'):
  assert p.read_bytes()==(ROOT/'src/baseline'/p.name).read_bytes(),p
 assert (base/'source/prelaunch/lease.json').read_bytes()==(pre/'lease.json').read_bytes()
 assert (base/'source/prelaunch/source-inventory.json').read_bytes()==(pre/'source-inventory.json').read_bytes()
 assert json.loads((base/'full-pass/resolved_config.json').read_text())==json.loads((run/'config.json').read_text())
 assert json.loads((base/'full-pass/manifest.json').read_text())['config_sha256']==sha(run/'config.json')
 result=subprocess.run([sys.executable,str(run/'03_verify.py'),str(base/'full-pass')],
  capture_output=True,text=True,encoding='utf-8',timeout=600)
 assert result.returncode==0,result.stderr
 scientific=json.loads(result.stdout);assert scientific['verified'] and scientific['temporal_verified']
 # Independently hash retained initial tensors in the recorded unique parameter order.
 from safetensors import safe_open
 cfg=json.loads((run/'config.json').read_text())
 metadata=json.loads((base/'full-pass/temporal/sketch_metadata.json').read_text())
 initial_digest=hashlib.sha256(); parameter_count=0
 with safe_open(base/'full-pass/checkpoints/initial/model/model.safetensors',framework='pt',device='cpu') as initial:
  for entry in metadata['scope']:
   tensor=initial.get_tensor(entry['name'])
   assert list(tensor.shape)==entry['shape'] and str(tensor.dtype)=='torch.float32'
   initial_digest.update(entry['name'].encode()+b'\0'); initial_digest.update(tensor.numpy().tobytes())
   parameter_count+=tensor.numel()
 assert parameter_count==cfg['model']['parameter_count']
 assert initial_digest.hexdigest()==cfg['model']['initial_parameter_sha256']
 row=next(r for r in json.loads((HERE/'cohort.json').read_text()) if r['run']==run_name)
 identity=json.loads((base/'full-pass/input_identity.json').read_text())
 reference=ROOT/'runs'/row['reference']/'artifacts/attempt-01/full-pass'
 assert identity['order_sha256']==json.loads((reference/'input_identity.json').read_text())['order_sha256']
 receipt={'pod_id':lease['pod_id'],'attempt':attempt,'archive_bytes':expected_bytes,
  'archive_sha256':expected_sha,**inventory,'scientific_verification':scientific,
  'frozen_sources_match':True,'reference_order_matches':True,
  'resolved_config_semantically_matches':True,'frozen_input_config_sha256':sha(run/'config.json'),
  'resolved_config_sha256':sha(base/'full-pass/resolved_config.json'),
  'verified_initial_checkpoint_parameter_sha256':initial_digest.hexdigest(),
  'accepted_at':datetime.now(timezone.utc).isoformat()}
 (pre/f'{attempt}-retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':accept(sys.argv[1])
