"""Compress remaining immutable recovery files on container disks; verify locally."""
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


def scopes(record):
    return ["full-pass", "calibration", "calibration-shared-gpu"] + (["calibration-single-job"] if record["run_id"] == "024" else [])


def prepare(record):
    parent = ROOT / "runs" / record["run"] / "artifacts/attempt-01"
    accepted = {}
    for scope in scopes(record):
        inventory_path = parent / (scope + "-remote-transfer-inventory.json")
        if inventory_path.exists():
            for item in json.loads(inventory_path.read_text())["files"]:
                path = parent / scope / item["path"]
                if path.exists() and path.stat().st_size == item["bytes"]:
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]:
                            accepted[scope + "/" + item["path"]] = item["sha256"]
    task = {"run": record["run"], "run_id": record["run_id"], "scopes": scopes(record), "accepted": accepted}
    code = """from pathlib import Path
import hashlib,json,tarfile,time
task=json.loads(TASK); root=Path('/workspace/intermediate-lr'); parent=root/'runs'/task['run']/'artifacts/attempt-01'
archive=Path('/opt/intermediate-lr-export-'+task['run_id']+'.tar.gz'); ready=archive.with_suffix('.ready.json')
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for scope in task['scopes']:
  folder=parent/scope; assert json.loads((folder/'verification.json').read_text())['verified']
  inventory=json.loads((folder/'transfer_inventory.json').read_text())
  for item in inventory['files']:
   key=scope+'/'+item['path']
   if task['accepted'].get(key)==item['sha256']: continue
   p=folder/item['path']; assert p.stat().st_size==item['bytes']; tar.add(p,arcname=key)
  tar.add(folder/'transfer_inventory.json',arcname=scope+'/transfer_inventory.json')
with archive.open('rb') as stream: sha=hashlib.file_digest(stream,'sha256').hexdigest()
ready.write_text(json.dumps({'path':str(archive),'sha256':sha,'bytes':archive.stat().st_size,'ready_epoch':time.time()}))
""".replace("TASK", repr(json.dumps(task)))
    command = "nohup setsid --fork python3 -c " + shlex.quote(code) + " > " + operate.ANALYSIS + "/bundle-" + record["run_id"] + ".log 2>&1 < /dev/null & echo $!"
    pid = int(operate.ssh(command, run_id=record["run_id"]).strip())
    operate.save(ROOT / "runs" / record["run"] / "prelaunch/bundle-preparation.json", {"pid": pid, "prepared_epoch": time.time(), "already_verified_files": accepted, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"run": record["run_id"], "bundle_pid": pid, "verified_files_skipped": len(accepted)}), flush=True)


def retrieve(record):
    run = ROOT / "runs" / record["run"]
    if all((run / "prelaunch" / ("retrieval-receipt.json" if scope == "full-pass" else scope + "-retrieval-receipt.json")).exists() for scope in scopes(record)):
        return
    ready = "/opt/intermediate-lr-export-" + record["run_id"] + ".tar.ready.json"
    proof = json.loads(operate.ssh("cat " + ready, run_id=record["run_id"]))
    t = operate.transport(record["run_id"])
    target = ROOT / "cache/runpod" / ("009-export-" + record["run_id"] + ".tar.gz")
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    if not target.exists() or target.stat().st_size != proof["bytes"] or digest(target) != proof["sha256"]:
        operate.scp(t["user"] + "@" + t["host"] + ":" + proof["path"], str(target), run_id=record["run_id"])
    assert target.stat().st_size == proof["bytes"]
    with target.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == proof["sha256"]
    parent = run / "artifacts/attempt-01"
    with tarfile.open(target) as tar:
        for member in tar:
            path = (parent / member.name).resolve()
            assert member.isfile() and path.is_relative_to(parent.resolve()) and Path(member.name).parts[0] in scopes(record)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("wb") as destination:
                __import__("shutil").copyfileobj(source, destination, 1024 * 1024)
    for scope in scopes(record):
        folder = parent / scope
        inventory_path = folder / "transfer_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        if scope == "full-pass":
            metric_inventory = HERE / "scratch/completed-metrics" / record["run_id"] / "transfer_inventory.json"
            assert digest(inventory_path) == digest(metric_inventory)
        for item in inventory["files"]:
            path = (folder / item["path"]).resolve()
            assert path.is_relative_to(folder.resolve()) and path.stat().st_size == item["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
        assert json.loads((folder / "verification.json").read_text())["verified"]
        receipt = "retrieval-receipt.json" if scope == "full-pass" else scope + "-retrieval-receipt.json"
        operate.save(run / "prelaunch" / receipt, {"accepted": True, "pod_id": t["pod_id"], "files_verified": len(inventory["files"]), "bytes": sum(i["bytes"] for i in inventory["files"]), "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(), "retrieved_epoch": time.time(), "bundle_sha256": proof["sha256"]})


if __name__ == "__main__":
    for record in operate.records():
        prepare(record)
