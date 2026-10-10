"""Native SSH/SCP for the explicitly attributed Analysis009 worker."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = "/workspace/intermediate-lr"
ANALYSIS = REMOTE + "/analyses/" + HERE.name
PYTHON = "/opt/intermediate-lr-env/bin/python"
DEVICES = {"024": 0, "025": 1, "026": 0, "027": 1}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def transport(run_id=None):
    filename = "transport-secondary.json" if run_id in ("026", "027") else "transport.json"
    t = json.loads((HERE / filename).read_text())
    owned = json.loads((HERE / "created-resources.json").read_text())
    assert t["pod_id"] in owned.get("pod_ids", [owned["pod_id"]])
    return t


def ssh(command, timeout=90, run_id=None):
    t = transport(run_id)
    common = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
              "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
              "-o", "StrictHostKeyChecking=accept-new", "-o",
              "UserKnownHostsFile=" + str(ROOT / "cache/runpod/known_hosts"),
              "-i", "C:/Users/thima/.ssh/id_ed25519_runpod"]
    return subprocess.run(["ssh", *common, "-p", str(t["port"]), t["user"] + "@" + t["host"], command],
                          check=True, capture_output=True, text=True, encoding="utf-8", timeout=timeout).stdout


def scp(source, destination, timeout=2700, recursive=False, run_id=None):
    t = transport(run_id)
    command = ["scp", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
               "-o", "StrictHostKeyChecking=accept-new", "-o",
               "UserKnownHostsFile=" + str(ROOT / "cache/runpod/known_hosts"),
               "-i", "C:/Users/thima/.ssh/id_ed25519_runpod", "-P", str(t["port"])]
    if recursive:
        command.append("-r")
    subprocess.run([*command, source, destination], check=True, timeout=timeout)


def records():
    return json.loads((HERE / "cohort.json").read_text())["runs"]


def snapshot():
    code = """from pathlib import Path
import json,subprocess,time
root=Path('/workspace/intermediate-lr')
rows=[]
for r in json.loads((root/'analyses/009-2026-10-09-intermediate-lr/cohort.json').read_text())['runs']:
    run=root/'runs'/r['run']; item={'run_id':r['run_id'],'peak_lr':r['peak_lr']}
    for scope in ('calibration','full-pass'):
        folder=run/'artifacts/attempt-01'/scope; result={}
        for filename in ('manifest.json','summary.json','verification.json'):
            path=folder/filename
            if path.exists(): result[filename[:-5]]=json.loads(path.read_text())
        events=folder/'events.jsonl'
        if events.exists():
            updates=[]
            for line in events.read_text().splitlines():
                try: e=json.loads(line)
                except json.JSONDecodeError: continue
                if e.get('event')=='update': updates.append(e)
            result['recent_updates']=updates[-30:]
            result['event_age_seconds']=time.time()-events.stat().st_mtime
        state=run/'artifacts/attempt-01'/(scope+'-state.json')
        if state.exists(): result['wrapper']=json.loads(state.read_text())
        item[scope]=result
    rows.append(item)
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout
usage=subprocess.run(['du','-sb','/workspace'],check=True,capture_output=True,text=True).stdout
print(json.dumps({'epoch':time.time(),'runs':rows,'gpu':gpu,'network_used_bytes':int(usage.split()[0]),'network_quota_gb':60,'disk':subprocess.run(['df','-h','/workspace','/opt'],capture_output=True,text=True).stdout}))
"""
    result = json.loads(ssh("python3 -c " + shlex.quote(code)))
    if (HERE / "transport-secondary.json").exists():
        result["gpu_secondary"] = ssh("nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader", run_id="026")
    return result


def launch(calibrate):
    for record in records():
        rid = record["run_id"]
        run = ROOT / "runs" / record["run"]
        receipt = run / "prelaunch" / ("calibration-launch.json" if calibrate else "launch.json")
        if receipt.exists():
            raise RuntimeError("Launch already recorded; do not duplicate execution.")
        if not calibrate:
            assert json.loads((run / "prelaunch/calibration-acceptance.json").read_text())["qualified"]
        command = f"CUDA_VISIBLE_DEVICES={DEVICES[rid]} TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 nohup setsid --fork {PYTHON} {ANALYSIS}/01_worker.py {rid}"
        if calibrate:
            command += " --calibrate"
        scope = "calibration" if calibrate else "full-pass"
        command += f" > {ANALYSIS}/run{rid}-{scope}.log 2>&1 < /dev/null & echo $!"
        pid = int(ssh(command, run_id=rid).strip())
        save(receipt, {"pod_id": transport(rid)["pod_id"], "pid": pid, "device": DEVICES[rid],
                       "scope": scope, "launched_epoch": time.time()})
        print(json.dumps({"run_id": rid, "scope": scope, "pid": pid}), flush=True)


def qualify():
    state = snapshot()
    remaining = json.loads((HERE / "lease.json").read_text())["deadline_epoch"] - time.time()
    for record, row in zip(records(), state["runs"]):
        data = row["calibration"]
        assert data.get("wrapper", {}).get("status") == "completed", row["run_id"]
        assert data["verification"]["verified"]
        updates = [e for e in data["recent_updates"] if e["update"] >= 9]
        assert len(updates) == 4
        mean = statistics.mean(e["update_seconds"] for e in updates)
        projected = mean * 712 + 2 * data["summary"]["final_validation"]["seconds"] + 240 + 3600
        proof = {"qualified": projected < remaining, "calibration_updates": 12,
                 "plateau_sample_updates": [9, 10, 11, 12], "seconds_per_update": mean,
                 "input_tokens_per_second": 2097152 / mean, "projected_total_seconds": projected,
                 "lease_remaining_seconds": remaining, "checkpoint_allowance_seconds": 240,
                 "retrieval_allowance_seconds": 3600, "concurrency": "four_processes_one_per_gpu_two_workers",
                 "calibration_scientifically_verified": True, "source_packet_sha256":
                 json.loads((HERE / "packet_identity.json").read_text())["sha256"]}
        path = ROOT / "runs" / record["run"] / "prelaunch/calibration-acceptance.json"
        save(path, proof)
        t = transport(record["run_id"])
        scp(str(path), t["user"] + "@" + t["host"] + ":" +
            REMOTE + "/runs/" + record["run"] + "/prelaunch/calibration-acceptance.json", run_id=record["run_id"])
        print(json.dumps({"run_id": record["run_id"], **proof}), flush=True)
    assert all(json.loads((ROOT / "runs" / r["run"] / "prelaunch/calibration-acceptance.json").read_text())["qualified"]
               for r in records()), "Measured ETC exceeds current lease; do not launch."


def retrieve_folder(record, scope):
    run = ROOT / "runs" / record["run"]
    relative = "runs/" + record["run"] + "/artifacts/attempt-01/" + scope
    inventory_sha = ssh(f"sha256sum {REMOTE}/{relative}/transfer_inventory.json", run_id=record["run_id"]).split()[0]
    parent = run / "artifacts/attempt-01"
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / scope
    # Resume file-by-file: a connection failure never requires copying accepted large states again.
    t = transport(record["run_id"])
    inventory_path = parent / (scope + "-remote-transfer-inventory.json")
    scp(t["user"] + "@" + t["host"] + ":" + REMOTE + "/" + relative + "/transfer_inventory.json", str(inventory_path), run_id=record["run_id"])
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == inventory_sha
    inventory = json.loads(inventory_path.read_text())
    for item in inventory["files"]:
        path = (folder / item["path"]).resolve()
        assert path.is_relative_to(folder.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == item["bytes"]:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]:
                    continue
        scp(t["user"] + "@" + t["host"] + ":" + REMOTE + "/" + relative + "/" + item["path"], str(path), run_id=record["run_id"])
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    (folder / "transfer_inventory.json").write_bytes(inventory_path.read_bytes())
    assert json.loads((folder / "verification.json").read_text())["verified"]
    return {"accepted": True, "pod_id": t["pod_id"], "files_verified": len(inventory["files"]),
            "bytes": sum(i["bytes"] for i in inventory["files"]), "inventory_sha256": inventory_sha,
            "retrieved_epoch": time.time()}


def retrieve(record):
    receipt = ROOT / "runs" / record["run"] / "prelaunch/retrieval-receipt.json"
    if not (receipt.exists() and json.loads(receipt.read_text())["accepted"]):
        save(receipt, retrieve_folder(record, "full-pass"))


def retrieve_calibrations(record):
    run = ROOT / "runs" / record["run"]
    scopes = ["calibration", "calibration-shared-gpu"]
    if record["run_id"] == "024":
        scopes.append("calibration-single-job")
    for scope in scopes:
        receipt = run / "prelaunch" / (scope + "-retrieval-receipt.json")
        if not (receipt.exists() and json.loads(receipt.read_text())["accepted"]):
            save(receipt, retrieve_folder(record, scope))


def retrieve_execution_records():
    # Read only this cohort's wrapper/setup records; retain them before volume cleanup.
    code = """from pathlib import Path
import hashlib,json
root=Path('/workspace/intermediate-lr'); analysis=root/'analyses/009-2026-10-09-intermediate-lr'
files=list(analysis.glob('*.log'))+list(analysis.glob('environment-lock-*.txt'))
for r in json.loads((analysis/'cohort.json').read_text())['runs']:
    folder=root/'runs'/r['run']/'artifacts/attempt-01'
    files.extend(folder.glob('*-state.json')); files.extend(folder.glob('*.log'))
print(json.dumps([{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files)) if p.is_file()]))
"""
    inventory = json.loads(ssh("python3 -c " + shlex.quote(code)))
    t = transport()
    for item in inventory:
        path = (ROOT / item["path"]).resolve()
        assert path.is_relative_to(ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        scp(t["user"] + "@" + t["host"] + ":" + REMOTE + "/" + item["path"], str(path))
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    save(HERE / "execution-records-retrieval.json", {"accepted": True, "files": inventory, "retrieved_epoch": time.time()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("probe", "upload-source", "bootstrap", "upload-data", "stage", "setup-status",
                                    "calibrate", "qualify", "full", "status", "retrieve"))
    args = p.parse_args()
    t = transport()
    destination = t["user"] + "@" + t["host"]
    if args.action == "probe":
        print(ssh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; df -h /workspace /opt"))
    elif args.action == "upload-source":
        scp(str(ROOT / "cache/runpod/intermediate-lr-source.tar.gz"), destination + ":/workspace/intermediate-lr-source.tar.gz")
    elif args.action == "bootstrap":
        expected = json.loads((HERE / "packet_identity.json").read_text())["sha256"]
        print(ssh(f"echo '{expected}  /workspace/intermediate-lr-source.tar.gz' | sha256sum -c - && mkdir -p {REMOTE} && tar -xzf /workspace/intermediate-lr-source.tar.gz -C {REMOTE}"))
        # Install can run while the large token file is being transferred.
        print(ssh(f"setsid bash {ANALYSIS}/03_setup.sh > {ANALYSIS}/setup.log 2>&1 < /dev/null & echo $!"))
    elif args.action == "upload-data":
        scp(str(ROOT / "cache/runpod/train-tokens.gz"), destination + ":/workspace/intermediate-lr-train.gz")
    elif args.action == "stage":
        command = f"gzip -dc /workspace/intermediate-lr-train.gz > {REMOTE}/data/train/tokens.int32.bin.tmp && mv {REMOTE}/data/train/tokens.int32.bin.tmp {REMOTE}/data/train/tokens.int32.bin"
        print(ssh("setsid bash -c " + shlex.quote(command) + f" > {ANALYSIS}/stage.log 2>&1 < /dev/null & echo $!"))
    elif args.action == "setup-status":
        print(ssh(f"tail -n 6 {ANALYSIS}/setup.log; tail -n 6 {ANALYSIS}/stage.log 2>/dev/null || true"))
    elif args.action in ("calibrate", "full"):
        launch(args.action == "calibrate")
    elif args.action == "qualify":
        qualify()
    elif args.action == "status":
        state = snapshot()
        save(HERE / "latest-status.json", state)
        print(json.dumps(state))
    elif args.action == "retrieve":
        for record in records():
            retrieve(record)
            print(json.dumps({"run_id": record["run_id"], "retrieved": True}), flush=True)


if __name__ == "__main__":
    main()
