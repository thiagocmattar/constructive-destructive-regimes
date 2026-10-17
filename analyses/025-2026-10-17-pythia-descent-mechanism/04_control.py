"""Local Runpod monitoring/retrieval and an independent task-owned spending guard."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tools.artifact_inventory import verify

CLI = ROOT / "tools/bin/runpodctl.exe"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cli(*args, check=True):
    result = subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=45)
    if check and result.returncode:
        raise RuntimeError(result.stderr)
    return result


def confirm_absent(*args):
    result = cli(*args, check=False)
    if result.returncode == 0:
        raise RuntimeError("Resource still exists after deletion")
    try:
        error = json.loads(result.stderr.splitlines()[0])
    except (ValueError, IndexError):
        raise RuntimeError("Could not confirm resource deletion")
    if error.get("code") != "not_found":
        raise RuntimeError("Deletion read failed: " + result.stderr)


def transport(pod, lease, scp=False):
    return ["scp" if scp else "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(ROOT / "cache/runpod/known_hosts"),
            "-i", lease["ssh_key"], "-P" if scp else "-p", str(pod["port"])]


def remote_status(pod, lease):
    script = '''import json,time,shutil
from pathlib import Path
root=Path('/workspace/constructive')
record=json.loads((root/'worker-control/worker-WORKER.json').read_text())
if record.get('output'):
 p=Path(record['output'])/'manifest.json'
 if p.exists():record['manifest']=json.loads(p.read_text())
 events=Path(record['output'])/'events.jsonl'
 if events.exists():
  with events.open('rb') as stream:
   stream.seek(0,2); size=stream.tell(); stream.seek(max(0,size-65536)); lines=stream.read().decode().splitlines()
  for line in reversed(lines):
   try:event=json.loads(line)
   except ValueError:continue
   if event.get('event')=='update':record['latest_update']=event; break
record['disk_free_gb']=shutil.disk_usage(root).free/1e9
calibration=root/'worker-control/calibration-WORKER.json'
if calibration.exists():record['calibration']=json.loads(calibration.read_text())
record['observed_at']=time.time()
print(json.dumps(record))
'''.replace("WORKER", str(pod["worker"]))
    result = subprocess.run(transport(pod, lease) + [f"{pod['user']}@{pod['host']}", "python3 -"],
        input=script, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def copy_file(pod, lease, relative, destination):
    remote = "/workspace/constructive/" + relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(transport(pod, lease, scp=True) + ["-q",
        f"{pod['user']}@{pod['host']}:{remote}", str(destination)], capture_output=True, text=True, timeout=2700)
    if result.returncode:
        raise RuntimeError(result.stderr)


def retrieve(pod, lease, name, accepted):
    relative = "runs/" + name + "/artifacts/attempt-01/full-pass/"
    output = ROOT / relative
    copy_file(pod, lease, relative + "inventory.json", output / "inventory.json")
    inventory = json.loads((output / "inventory.json").read_text())
    for item in inventory["files"]:
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in str(path) or ":" in str(path):
            raise ValueError("Unsafe artifact inventory path")
        destination = output / str(path)
        if destination.is_file() and destination.stat().st_size == item["bytes"] and sha(destination) == item["sha256"]:
            continue
        partial = destination.with_name(destination.name + ".part")
        copy_file(pod, lease, relative + str(path), partial)
        if partial.stat().st_size != item["bytes"] or sha(partial) != item["sha256"]:
            raise ValueError("Downloaded artifact differs: " + str(path))
        partial.replace(destination)
    receipt = verify(output, output / "inventory.json", strict=True)
    if not receipt["verified"]:
        raise ValueError("Local inventory verification failed")
    spec = importlib.util.spec_from_file_location("h021_verify", HERE / "03_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt["scientific_verification"] = module.verify(output)
    receipt["inventory_sha256"] = sha(output / "inventory.json")
    receipt["accepted_at_epoch"] = time.time()
    save(ROOT / "runs" / name / "prelaunch/attempt-01-retrieval.json", receipt)
    accepted.append(name)
    save(HERE / "retrieved.json", dict(runs=accepted))


def assert_owned(pod):
    data = json.loads(cli("pod", "get", pod["id"]).stdout)
    if data["id"] != pod["id"] or data["name"] != pod["name"] or not data["name"].startswith("cdr-analysis025-"):
        raise ValueError("Refusing an unattributed pod action")
    return data


def guard(lease):
    authorization = json.loads((HERE / "authorization.json").read_text())
    hourly = sum(p["hourly_price"] for p in lease["pods"]) + lease["storage_hourly_price"]
    cutoff = min(lease["hard_deadline_epoch"], lease["created_at_epoch"] + authorization["maximum_total_usd"] / hourly * 3600)
    while time.time() < cutoff:
        if (HERE / "closeout.json").exists():
            return
        time.sleep(10)
    outcomes = []
    for pod in lease["pods"]:
        for attempt in range(3):
            try:
                data = assert_owned(pod)
                if data.get("runtimeStatus") not in {"stopped", "terminated"}:
                    cli("pod", "stop", pod["id"])
                after = json.loads(cli("pod", "get", pod["id"]).stdout)
                outcomes.append(dict(pod=pod["id"], runtimeStatus=after.get("runtimeStatus"), status=after.get("desiredStatus")))
                break
            except Exception as error:
                outcomes.append(dict(pod=pod["id"], attempt=attempt, error=repr(error)))
                time.sleep(5)
    save(HERE / "deadline_stop.json", dict(at_epoch=time.time(), outcomes=outcomes))


def monitor(lease, once):
    accepted = json.loads((HERE / "retrieved.json").read_text())["runs"] if (HERE / "retrieved.json").exists() else []
    expected = {r["run"] for r in json.loads((HERE / "cohort.json").read_text())}
    while True:
        snapshots = []
        for pod in lease["pods"]:
            try:
                snapshot = remote_status(pod, lease)
                snapshots.append(dict(pod=pod["id"], **snapshot))
                if not once:
                    for name in snapshot["completed"]:
                        if name not in expected:
                            raise ValueError("Worker returned an unknown run")
                        if name not in accepted:
                            retrieve(pod, lease, name, accepted)
                    if snapshot["status"] == "failed":
                        assert_owned(pod)
                        cli("pod", "stop", pod["id"])
            except Exception as error:
                snapshots.append(dict(pod=pod["id"], error=repr(error)))
        save(HERE / "latest-status.json", dict(observed_at=time.time(), pods=snapshots, retrieved=len(accepted)))
        if once or set(accepted) == expected:
            return
        if time.time() >= lease["hard_deadline_epoch"]:
            return
        time.sleep(300)


def cleanup(lease):
    expected = {r["run"] for r in json.loads((HERE / "cohort.json").read_text())}
    assert set(json.loads((HERE / "retrieved.json").read_text())["runs"]) == expected
    for name in expected:
        receipt = json.loads((ROOT / "runs" / name / "prelaunch/attempt-01-retrieval.json").read_text())
        assert receipt["verified"] and receipt["scientific_verification"]["verified"]
    for pod in lease["pods"]:
        assert_owned(pod)
        cli("pod", "delete", pod["id"], check=False)
        confirm_absent("pod", "get", pod["id"])
    volume = json.loads(cli("network-volume", "get", lease["volume"]["id"]).stdout)
    assert volume["id"] == lease["volume"]["id"]
    assert volume["name"] == lease["volume"]["name"] == "cdr-analysis025-descent"
    cli("network-volume", "delete", lease["volume"]["id"], check=False)
    confirm_absent("network-volume", "get", lease["volume"]["id"])
    save(HERE / "closeout.json", dict(completed_at_epoch=time.time(), task_compute_deleted=True,
                                     task_storage_deleted=True, billing="provisional until provider reconciliation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["guard", "monitor", "status", "cleanup"])
    args = parser.parse_args()
    lease = json.loads((HERE / "lease.json").read_text())
    if lease["owner"] != "Analysis025-current-task":
        raise ValueError("Unattributed lease")
    if args.action == "guard":
        guard(lease)
    elif args.action in {"monitor", "status"}:
        monitor(lease, args.action == "status")
        if args.action == "monitor":
            expected = {r["run"] for r in json.loads((HERE / "cohort.json").read_text())}
            retrieved = json.loads((HERE / "retrieved.json").read_text())["runs"] if (HERE / "retrieved.json").exists() else []
            if set(retrieved) == expected:
                cleanup(lease)
    else:
        cleanup(lease)


if __name__ == "__main__":
    main()
