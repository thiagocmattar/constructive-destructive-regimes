"""Detached, bounded sequential execution of the three authorized Spark conditions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from baseline.training import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    cohort = json.loads((HERE / "cohort.json").read_text())
    inventory = json.loads((HERE / "source_inventory.json").read_text())
    for item in inventory["files"]:
        path = ROOT / item["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Executed source differs: {item['path']}")
    state_path = HERE / ("calibration_queue.json" if args.calibrate else "queue_state.json")
    if state_path.exists():
        raise ValueError("Preserve the existing attempt instead of relaunching it.")
    state = {"status": "running", "scope": "calibration" if args.calibrate else "full_pass",
             "started_at": now(), "current_run": None,
             "runs": {r["run_id"]: {"status": "queued", "peak_lr": r["peak_lr"]} for r in cohort["runs"]}}
    write_json(state_path, state)
    for record in cohort["runs"]:
        rid = record["run_id"]
        run = ROOT / "runs" / record["run"]
        attempt = run / "artifacts/attempt-01"
        attempt.mkdir(parents=True, exist_ok=True)
        output = attempt / ("calibration" if args.calibrate else "full-pass")
        if not args.calibrate:
            proof = json.loads((attempt / "calibration/verification.json").read_text())
            if not proof["verified"] or proof["scope"] != "smoke_calibration":
                raise ValueError("Require successful independent calibration before launch.")
        state["current_run"] = rid
        state["runs"][rid].update(status="running", started_at=now())
        write_json(state_path, state)
        command = [sys.executable, str(run / "01_train.py"), "--config", str(run / "config.json"),
                   "--cache", str(ROOT / "data"), "--output", str(output)]
        if args.calibrate:
            command += ["--calibrate-updates", "3"]
        try:
            with (attempt / ("calibration.log" if args.calibrate else "full-pass.log")).open("x") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=900 if args.calibrate else 8 * 3600, check=False)
            if completed.returncode:
                raise RuntimeError(f"Training exited {completed.returncode}")
            subprocess.run([sys.executable, str(HERE / "02_verify.py"), str(output)],
                           cwd=ROOT, timeout=600, check=True)
            state["runs"][rid].update(status="completed", finished_at=now())
        except BaseException as error:
            state["runs"][rid].update(status="failed", finished_at=now(), error=str(error))
            if (output / "manifest.json").exists():
                manifest = json.loads((output / "manifest.json").read_text())
                if manifest["status"] not in ("failed", "completed"):
                    manifest.update(status="failed", finished_at=now(), error=str(error))
                    write_json(output / "manifest.json", manifest)
            print(traceback.format_exc(), flush=True)
        write_json(state_path, state)
    state["status"] = ("completed" if all(v["status"] == "completed" for v in state["runs"].values())
                       else "completed_with_failures")
    state.update(current_run=None, finished_at=now())
    write_json(state_path, state)


if __name__ == "__main__":
    main()
