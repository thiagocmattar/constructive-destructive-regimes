"""Verify the owned worker's entire archive before resource deletion."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import tarfile
import torch

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs'/sys.argv[1]
rid=run.name[:3]
archive=ROOT/f'cache/runpod/run{rid}-attempt-01.tar.gz'
expected_bytes=int(archive.with_name(archive.name+'.bytes').read_text().strip())
expected_hash=archive.with_name(archive.name+'.sha256').read_text().split()[0]
assert archive.stat().st_size==expected_bytes
with archive.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected_hash
target=(run/'artifacts').resolve()
assert target.is_relative_to(ROOT.resolve()) and not (target/'attempt-01').exists()
with tarfile.open(archive,'r:gz') as incoming:
    members=incoming.getmembers()
    seen=set()
    for m in members:
        p=PurePosixPath(m.name)
        assert p.parts and p.parts[0]=='attempt-01' and not p.is_absolute() and '..' not in p.parts
        assert '\\' not in m.name and ':' not in m.name and m.name not in seen
        assert m.isdir() or m.isfile()
        seen.add(m.name)
    incoming.extractall(target,members=members,filter='data')
sys.path.insert(0,str(ROOT/'tools'))
from artifact_inventory import verify
base=target/'attempt-01'
accepted=verify(base,base/'inventory.json',strict=True)
assert accepted['verified'],accepted
# Check execution source copies against the exact controller packet.
source_matches={}
for p in (base/'source').glob('*'):
    if p.is_file():
        other=run/p.name
        assert other.is_file() and p.read_bytes()==other.read_bytes(),p
        source_matches[str(p.relative_to(base))]=True
for p in (base/'source/baseline').glob('*.py'):
    assert p.read_bytes()==(ROOT/'src/baseline'/p.name).read_bytes(),p
    source_matches[str(p.relative_to(base))]=True
assert (base/'source/prelaunch/source-inventory.json').read_bytes()==(run/'prelaunch/source-inventory.json').read_bytes()
spec=importlib.util.spec_from_file_location('verify_scale_'+rid,run/'03_verify.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
scientific=module.verify(base/'full-pass')
assert scientific['verified']
from safetensors.torch import load_file
initial=load_file(base/'full-pass/checkpoints/initial/model/model.safetensors')
names=json.loads((base/'full-pass/gradient_state_identity.json').read_text())['parameter_names']
initial_digest=hashlib.sha256()
for name in names:
    p=initial[name]
    assert p.dtype==torch.float32
    initial_digest.update(name.encode()+b'\0')
    initial_digest.update(p.numpy().tobytes())
initial_sha=initial_digest.hexdigest()
cfg=json.loads((run/'config.json').read_text())
assert initial_sha==cfg['model']['initial_parameter_sha256']
assert sum(initial[n].numel() for n in names)==cfg['model']['parameter_count']
del initial
events=[e for line in (base/'full-pass/events.jsonl').read_text().splitlines()
        if (e:=json.loads(line))['event']=='update']
relative=[abs(e['gradient_norm']-e['gradient_clip_returned_norm'])/e['gradient_norm']
          for e in events if e['gradient_norm']]
assert max(relative)<1e-5
transport=json.loads((run/'prelaunch/transport.json').read_text())
receipt={'pod_id':transport['pod_id'],'attempt':'attempt-01',
    'archive_bytes':expected_bytes,'archive_sha256':actual,**accepted,
    'scientific_verification':scientific,'frozen_source_copies_match':source_matches,
    'initial_checkpoint_parameter_sha256':initial_sha,
    'max_raw_norm_relative_error_vs_independent_clip_norm':max(relative),
    'accepted_at':datetime.now(timezone.utc).isoformat()}
(run/'prelaunch/attempt1-retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='frozen_source_copies_match'},indent=2))
