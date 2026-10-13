"""Bounded sequential calibration that survives SSH disconnection when detached."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cifar_eos import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "started_utc": stamp(), "mode": "calibration only", "optimizers": {}, "timeout_seconds_per_optimizer": args.timeout_seconds}
    write_json(args.output / "queue.json", state)
    try:
        for optimizer in ("gd", "adam"):
            state["current_optimizer"] = optimizer
            write_json(args.output / "queue.json", state)
            command = [sys.executable, str(HERE / "01_experiment.py"), "--inputs", str(args.inputs),
                       "--output", str(args.output / optimizer), "--optimizer", optimizer, "--calibrate", "--device", args.device]
            with (args.output / (optimizer + ".log")).open("x", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout_seconds)
            state["optimizers"][optimizer] = "complete"
            write_json(args.output / "queue.json", state)
        command = [sys.executable, str(HERE / "02_verify.py"), str(args.output / "gd"), str(args.output / "adam"), "--output", str(args.output / "verification.json")]
        subprocess.run(command, check=True, timeout=120)
        state.update(status="complete", completed_utc=stamp(), current_optimizer=None)
    except Exception as error:
        state.update(status="failed", completed_utc=stamp(), error=repr(error))
        write_json(args.output / "queue.json", state)
        raise
    write_json(args.output / "queue.json", state)
    write_json(args.output / "transfer_inventory.json", {"files": [
        {"path": p.relative_to(args.output).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "transfer_inventory.json"]})
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
