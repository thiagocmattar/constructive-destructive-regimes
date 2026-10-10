"""Retrieve completed compact metrics independently of ongoing recovery downloads."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shlex
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("operate009", HERE / "04_operate.py")
operate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operate)


def main():
    code = """from pathlib import Path
import hashlib,json,tarfile
root=Path('/workspace/intermediate-lr'); analysis=root/'analyses/009-2026-10-09-intermediate-lr'
names=('environment.json','events.jsonl','gradient_state_identity.json','input_identity.json','manifest.json','resolved_config.json','summary.json','verification.json','transfer_inventory.json')
entries=[]; archive=analysis/'completed-metrics.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for r in json.loads((analysis/'cohort.json').read_text())['runs']:
  folder=root/'runs'/r['run']/'artifacts/attempt-01/full-pass'
  assert json.loads((folder/'manifest.json').read_text())['status']=='completed'
  assert json.loads((folder/'verification.json').read_text())['verified']
  inventory={i['path']:i for i in json.loads((folder/'transfer_inventory.json').read_text())['files']}
  for name in names:
   p=folder/name; data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest()
   if name!='transfer_inventory.json': assert inventory[name]['bytes']==len(data) and inventory[name]['sha256']==sha
   relative=r['run_id']+'/'+name; tar.add(p,arcname=relative)
   entries.append({'path':relative,'bytes':len(data),'sha256':sha})
print(json.dumps({'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':entries}))
"""
    proof = json.loads(operate.ssh("python3 -c " + shlex.quote(code)))
    target = ROOT / "cache/runpod/009-completed-metrics.tar.gz"
    t = operate.transport()
    operate.scp(t["user"] + "@" + t["host"] + ":" + operate.ANALYSIS + "/completed-metrics.tar.gz", str(target))
    assert target.stat().st_size == proof["bytes"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == proof["sha256"]
    folder = HERE / "scratch/completed-metrics"
    folder.mkdir(parents=True, exist_ok=True)
    expected = {i["path"]: i for i in proof["files"]}
    with tarfile.open(target) as tar:
        assert {m.name for m in tar.getmembers()} == set(expected)
        for member in tar.getmembers():
            path = (folder / member.name).resolve()
            assert member.isfile() and path.is_relative_to(folder.resolve())
            data = tar.extractfile(member).read()
            item = expected[member.name]
            assert len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    proof.update(accepted=True, retrieved_epoch=time.time(), coverage="all_completed_metric_files; recovery retrieval remains separate")
    operate.save(HERE / "metrics-retrieval-receipt.json", proof)
    print(json.dumps({"accepted": True, "files": len(expected), "archive_bytes": proof["bytes"]}))


if __name__ == "__main__":
    main()
