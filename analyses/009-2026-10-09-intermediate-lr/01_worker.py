"""One detached, deadline-bounded condition; calibrate independently before full run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id")
    p.add_argument("--calibrate", action="store_true")
    args = p.parse_args()
    cohort = json.loads((HERE / "cohort.json").read_text())
    record = next(r for r in cohort["runs"] if r["run_id"] == args.run_id)
    run = ROOT / "runs" / record["run"]
    for item in json.loads((HERE / "source_inventory.json").read_text())["files"]:
        path = ROOT / item["path"]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == item["sha256"] and path.stat().st_size == item["bytes"], item["path"]
    attempt = run / "artifacts/attempt-01"
    attempt.mkdir(parents=True, exist_ok=True)
    scope = "calibration" if args.calibrate else "full-pass"
    state_path = attempt / (scope + "-state.json")
    if state_path.exists():
        raise RuntimeError("Attempt already started; inspect and preserve it.")
    output = attempt / scope
    if not args.calibrate:
        proof = json.loads((attempt / "calibration/verification.json").read_text())
        assert proof["verified"] and proof["scope"] == "smoke_calibration"
        assert json.loads((run / "prelaunch/calibration-acceptance.json").read_text())["qualified"]
    lease = json.loads((run / "prelaunch/lease.json").read_text())
    remaining = int(lease["deadline_epoch"] - time.time() - 600)
    if remaining < 60:
        raise RuntimeError("Lease cannot accommodate training plus retrieval margin.")
    state = {"status": "running", "pid": os.getpid(), "started_epoch": time.time(),
             "scope": scope, "run_id": args.run_id}
    save(state_path, state)
    command = [sys.executable, str(run / "01_train.py"), "--config", str(run / "config.json"),
               "--cache", "/opt/intermediate-lr-data", "--output", str(output)]
    if args.calibrate:
        command += ["--calibrate-updates", "12"]
    try:
        with (attempt / (scope + ".log")).open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(900, remaining) if args.calibrate else remaining, check=True)
        subprocess.run([sys.executable, str(HERE / "02_verify.py"), str(output)],
                       cwd=ROOT, timeout=600, check=True)
        state.update(status="completed", finished_epoch=time.time())
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                     finished_epoch=time.time())
    save(state_path, state)
    print(json.dumps(state), flush=True)
    if state["status"] != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
