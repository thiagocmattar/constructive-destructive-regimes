"""Detached, deadline-bounded Runpod worker for a preassigned set of H021 families."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tools.artifact_inventory import create, verify


def save(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(command, log, deadline):
    remaining = deadline - time.time()
    if remaining < 60:
        raise TimeoutError("Training deadline reached")
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    authorization = json.loads((HERE / "authorization.json").read_text())
    if not authorization["approved"] or authorization["maximum_total_usd"] > 25:
        raise ValueError("Missing approval for the prepared envelope")
    queue = json.loads((HERE / "worker_plan.json").read_text())[args.worker]
    control = ROOT / "worker-control"
    control.mkdir(exist_ok=True)
    lock = control / f"worker-{args.worker}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    status = control / f"worker-{args.worker}.json"
    record = dict(status="checking_inputs", worker=args.worker, pid=os.getpid(), completed=[])
    save(status, record)
    try:
        for filename in ("source_inventory.json", "input_packet.json"):
            inventory = json.loads((HERE / filename).read_text())
            import hashlib
            for item in inventory["files"]:
                path = ROOT / item["path"]
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
                    raise ValueError(f"Frozen input differs: {item['path']}")
        first = ROOT / "runs" / queue[0]["baseline"]
        calibration = first / "artifacts/attempt-01/calibration"
        calibration.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(first / "02_train.py"), "--config", str(first / "config.json"),
                   "--cache", str(ROOT / "inputs/cache"), "--output", str(calibration),
                   "--calibrate-updates", "3", "--deadline-epoch", str(args.deadline_epoch)]
        record.update(status="calibrating", run=first.name, output=str(calibration))
        save(status, record)
        import hashlib
        save(first / "prelaunch/calibration-launch.json", dict(worker=args.worker, started_at=time.time(),
            source_inventory_sha256=hashlib.sha256((HERE / "source_inventory.json").read_bytes()).hexdigest()))
        run(command, calibration.parent / "calibration.log", args.deadline_epoch)
        samples = [json.loads(s) for s in (calibration / "events.jsonl").read_text().splitlines()]
        samples = [s for s in samples if s["event"] == "update"][-2:]
        native = max(s["training_update_seconds"] for s in samples)
        probe = max(s["probe_and_snapshot_seconds"] for s in samples)
        historical = json.loads((HERE / "historical_timing.json").read_text())
        ratios = {r["size"]: r["mean_update_seconds"] for r in historical if r["seed"] == 1234}
        calibrated_size = queue[0]["size"]
        projection = sum(450 * native * ratios[f["size"]] / ratios[calibrated_size] +
                         13 * probe + 180 for f in queue)
        import torch
        memory = max(s["peak_cuda_allocated_bytes"] for s in samples)
        memory_limit = .95 * torch.cuda.get_device_properties(0).total_memory
        trial_rows = [json.loads(s) for s in (calibration / "probe_events.jsonl").read_text().splitlines()]
        controls = [r for r in trial_rows if r["candidate"] == "adam"]
        numerics = bool(controls) and all(abs(r["native_reconstruction_loss_gap"]) < 1e-5 and
            r["native_reconstruction_max_parameter_error"] < 1e-6 for r in controls)
        qualification = dict(training_update_seconds_upper=native, probe_overhead_seconds_upper=probe,
            projected_remaining_seconds=projection, measured_peak_cuda_bytes=memory,
            smaller_model_timing="qualified historical RTX4090 ratios to measured largest-model native time",
            memory_limit_bytes=memory_limit, native_reconstruction_controls_passed=numerics,
            qualified=numerics and memory < memory_limit and time.time() + 1.3 * projection < args.deadline_epoch)
        save(control / f"calibration-{args.worker}.json", qualification)
        if not qualification["qualified"]:
            raise RuntimeError("Measured memory/ETC exceeds the prepared envelope; main queue not started")
        for family in queue:
            for name in [family["baseline"], *family["branches"]]:
                directory = ROOT / "runs" / name
                output = directory / "artifacts/attempt-01/full-pass"
                output.parent.mkdir(parents=True, exist_ok=True)
                record.update(status="running", run=name, output=str(output))
                save(status, record)
                save(directory / "prelaunch/production-launch.json", dict(worker=args.worker, started_at=time.time(),
                    config_sha256=hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
                    source_inventory_sha256=hashlib.sha256((HERE / "source_inventory.json").read_bytes()).hexdigest()))
                command = [sys.executable, str(directory / "02_train.py"), "--config", str(directory / "config.json"),
                           "--cache", str(ROOT / "inputs/cache"), "--output", str(output),
                           "--deadline-epoch", str(args.deadline_epoch)]
                run(command, output.parent / "training.log", args.deadline_epoch)
                for path in [HERE / "03_verify.py", HERE / "source_inventory.json", HERE / "input_packet.json"]:
                    shutil.copy2(path, output / "source" / path.name)
                shutil.copy2(ROOT / f"environment-lock-worker-{args.worker}.txt", output / "environment-lock.txt")
                run([sys.executable, str(HERE / "03_verify.py"), str(output)],
                    output.parent / "verification.log", args.deadline_epoch)
                # Final recovery supersedes the rolling copy; retain both fork states.
                (output / "checkpoints/rolling.pt").unlink(missing_ok=True)
                inventory = output / "inventory.json"
                create(output, inventory)
                if not verify(output, inventory, strict=True)["verified"]:
                    raise RuntimeError("Remote inventory self-verification failed")
                record["completed"].append(name)
                save(status, record)
        record.update(status="completed", completed_at=time.time())
        save(status, record)
    except BaseException as exc:
        record.update(status="failed", error=repr(exc), failed_at=time.time())
        save(status, record)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
