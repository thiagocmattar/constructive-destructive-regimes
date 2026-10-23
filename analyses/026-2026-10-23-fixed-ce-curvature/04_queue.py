"""Detached, bounded sequential queue for exactly the frozen H022 cohort."""
import argparse
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import inventory, save_json, sha, stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--created-epoch", type=float, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    control = ROOT / "worker-control"
    control.mkdir(exist_ok=True)
    lock = control / "queue.lock"
    with lock.open("x") as stream:
        stream.write(str(time.time()))
    def terminate(signum, frame):
        raise TimeoutError("Independent deadline watchdog stopped the queue")
    signal.signal(signal.SIGTERM, terminate)
    status = {"status": "starting", "started": stamp(), "completed": [],
              "deadline_epoch": args.deadline_epoch}
    status_path = control / "status.json"
    save_json(status_path, status)
    try:
        frozen = json.loads((here / "source_inventory.json").read_text())
        for item in frozen["files"]:
            path = ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise RuntimeError("Source/input packet hash failed: " + item["path"])
        module_path = ROOT / "runs/154-2026-10-23-fixed-ce-calibration/02_worker.py"
        spec = importlib.util.spec_from_file_location("h022_worker", module_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        calibration_path = ROOT / "runs/154-2026-10-23-fixed-ce-calibration/artifacts/attempt-01"
        source = calibration_path / "source"
        source.mkdir(parents=True)
        import shutil
        for path in (module_path, ROOT / "src/fixed_ce_curvature/core.py", Path(__file__)):
            shutil.copy2(path, source / path.name)
        save_json(status_path, {**status, "status": "calibrating", "active": str(calibration_path.relative_to(ROOT))})
        worker.calibrate(calibration_path)
        calibration = json.loads((calibration_path / "calibration.json").read_text())
        required = calibration["with_30_percent_allowance_and_10min_retrieval_seconds"]
        available = args.deadline_epoch - time.time()
        save_json(control / "admission.json", {"utc": stamp(), "required_seconds": required,
                  "available_seconds": available, "admitted": required <= available,
                  "seconds_since_creation": time.time() - args.created_epoch})
        if required > available:
            status.update(status="calibration_exceeds_envelope", completed_at=stamp())
            save_json(status_path, status)
            return
        cohort = json.loads((here / "cohort.json").read_text())
        for entry in cohort:
            run = ROOT / "runs" / entry["run"]
            output = run / "artifacts/attempt-01"
            config = json.loads((run / "config.json").read_text())
            config["deadline_epoch"] = args.deadline_epoch
            status.update(status="training", active=entry, output=str(output.relative_to(ROOT)))
            save_json(status_path, status)
            worker.train(config, output)
            status["status"] = "diagnosing"
            save_json(status_path, status)
            worker.diagnose(output)
            manifest = json.loads((output / "manifest.json").read_text())
            manifest.update(status="complete", diagnosed_at=stamp())
            save_json(output / "manifest.json", manifest)
            proof = inventory(output)
            status["completed"].append({**entry, "output": str(output.relative_to(ROOT)),
                                        "bytes": proof["total_bytes"]})
            save_json(status_path, status)
            size = sum(p.stat().st_size for p in (ROOT / "runs").glob("*-2026-10-23-fixed-ce-*/artifacts/**/*") if p.is_file())
            if size > 5e9:
                raise RuntimeError("Five GB evidence allowance reached; required evidence preserved")
        status.update(status="complete", completed_at=stamp(), active=None)
        save_json(status_path, status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      failed_at=stamp())
        save_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
