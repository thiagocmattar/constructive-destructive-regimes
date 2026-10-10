"""Native SSH/SCP for this explicitly attributed six-job continuation cohort."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")


def native(row, action, command=None):
    args = [sys.executable, str(ROOT / "tools/scale_worker.py"), row["run"], action]
    if command is not None:
        args += ["--command", command]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=1900)
    if result.returncode:
        raise RuntimeError(f"{action}: {result.stderr[-400:]} {result.stdout[-400:]}")
    return result.stdout


def setup(row):
    run = ROOT / "runs" / row["run"]
    pre = run / "prelaunch"
    transport = json.loads((pre / "transport.json").read_text())
    assert transport["run_id"] == row["run"][:3]
    for action in ("probe", "upload-source", "bootstrap", "upload-data"):
        print(json.dumps({"run": row["run"][:3], "stage": action}), flush=True)
        result = native(row, action)
        if action == "probe":
            (pre / "hardware-probe.txt").write_text(result)
    common = ["scp", "-q", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile="+str(ROOT/"cache/runpod/known_hosts"),
        "-i", "C:/Users/thima/.ssh/id_ed25519_runpod", "-P", str(transport["port"])]
    print(json.dumps({"run": row["run"][:3], "stage": "upload-parent"}), flush=True)
    subprocess.run(common+[str(ROOT/row["parent_packet"]),
        transport["user"]+"@"+transport["host"]+":/workspace/parent.tar.gz"], check=True, timeout=1500)
    packet = next(p for p in json.loads((HERE/"parent_packets.json").read_text()) if p["size"] == row["size"])
    command = ("test $(stat -c %s /workspace/parent.tar.gz) -eq "+str(packet["bytes"])+
        " && echo "+shlex.quote(packet["sha256"]+"  /workspace/parent.tar.gz")+
        " | sha256sum -c - && mkdir /workspace/parent-checkpoint"+
        " && tar -xzf /workspace/parent.tar.gz -C /workspace/parent-checkpoint")
    native(row, "command", command)
    native(row, "stage")
    print(json.dumps({"run": row["run"][:3], "stage": "setup-and-staging-started"}), flush=True)


def work(row, action):
    pre = ROOT/"runs"/row["run"]/"prelaunch"
    if action == "retry-parent":
        transport = json.loads((pre/"transport.json").read_text())
        receipt = {"pod_id": transport["pod_id"], "started_at": datetime.now(timezone.utc).isoformat(),
            "reason": "Preflight SSH/SCP disconnect or extraction exceeded interactive timeout; unchanged scientific inputs"}
        (pre/"parent-transfer-retry.json").write_text(json.dumps(receipt, indent=2)+"\n")
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile="+str(ROOT/"cache/runpod/known_hosts"),
            "-i", "C:/Users/thima/.ssh/id_ed25519_runpod", "-P", str(transport["port"]),
            str(ROOT/row["parent_packet"]), transport["user"]+"@"+transport["host"]+":/workspace/parent.tar.gz"],
            check=True, timeout=1500)
        packet = next(p for p in json.loads((HERE/"parent_packets.json").read_text()) if p["size"] == row["size"])
        command = ("set -e; test $(stat -c %s /workspace/parent.tar.gz) -eq "+str(packet["bytes"])+
            "; echo "+shlex.quote(packet["sha256"]+"  /workspace/parent.tar.gz")+
            " | sha256sum -c -; mkdir -p /workspace/parent-checkpoint; tar -xzf /workspace/parent.tar.gz -C /workspace/parent-checkpoint; echo PARENT_READY")
        native(row, "command", "setsid bash -c "+shlex.quote(command)+" > /workspace/constructive/parent-retry.log 2>&1 < /dev/null & echo $!")
        native(row, "stage")
        print(json.dumps({"run": row["run"][:3], "stage": "parent-retry-and-staging-started"}), flush=True)
        return
    if action == "upload-lease":
        transport = json.loads((pre/"transport.json").read_text())
        destination = transport["user"]+"@"+transport["host"]+":/workspace/constructive/runs/"+row["run"]+"/prelaunch/lease.json"
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile="+str(ROOT/"cache/runpod/known_hosts"),
            "-i", "C:/Users/thima/.ssh/id_ed25519_runpod", "-P", str(transport["port"]),
            str(pre/"lease.json"), destination], check=True, timeout=60)
        print(json.dumps({"run": row["run"][:3], "stage": "lease-uploaded"}), flush=True)
        return
    if action == "setup":
        setup(row)
        return
    if action in ("calibrate", "full"):
        receipt = pre/("calibration-start.json" if action == "calibrate" else "full-pass-start.json")
        if receipt.exists():
            raise RuntimeError("Launch receipt exists; do not duplicate a running job.")
        if action == "full":
            assert json.loads((pre/"calibrated-launch.json").read_text())["qualified"]
        result = native(row, action)
        receipt.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
            "pid": int(result.strip()), "phase": action,
            "pod_id": json.loads((pre/"transport.json").read_text())["pod_id"]}, indent=2)+"\n")
    else:
        result = native(row, action)
    if action in ("status", "calibration-status"):
        snapshot = json.loads(result)
        m, e = snapshot.get("manifest", {}), snapshot.get("latest_update", {})
        result = {"status": m.get("status"), "update": m.get("completed_updates"),
            "loss": m.get("latest_loss"), "tokens": m.get("completed_input_tokens"),
            "tokens_per_second": e.get("tokens_per_second"),
            "remaining_training_seconds": m.get("remaining_training_seconds"),
            "peak_memory_gb": e.get("peak_cuda_allocated_bytes",0)/1e9,
            "overflow_retries": e.get("overflow_retries"), "verified": snapshot.get("verification",{}).get("verified")}
    print(json.dumps({"run": row["run"][:3], "action": action, "result": result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "retry-parent", "upload-lease", "setup-status", "verify-source", "calibrate",
        "calibration-status", "full", "status", "logs", "archive", "archive-status"])
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    rows = json.loads((HERE/"cohort.json").read_text())
    if args.ids:
        rows = [r for r in rows if r["run"][:3] in args.ids]
    assert rows
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda row: work(row,args.action),rows))


if __name__ == "__main__":
    main()
