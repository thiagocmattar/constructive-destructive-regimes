"""Bounded, detached sequential Spark production queue; preserve every attempt."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent


def inventory(output):
    write_json(output / "transfer_inventory.json", {"files": [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "transfer_inventory.json"]})


def mark_failed(output, error):
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.update(status="failed", ended_utc=stamp(), error=error)
        write_json(path, manifest)


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    for item in json.loads((HERE / "source_inventory.json").read_text())["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"], item["path"]
    assert file_hash(Path(cohort["remote_inputs"]) / "inputs.json") == cohort["input_manifest_sha256"]
    if (HERE / "queue_state.json").exists():
        raise FileExistsError("An executed queue exists; preserve it.")
    state = {"status": "running", "started_utc": stamp(), "current_run": None,
             "runs": {r["run_id"]: {"status": "queued", "lr": r["lr"], "optimizer": r["optimizer"]} for r in cohort["runs"]}}
    write_json(HERE / "queue_state.json", state)
    clock = time.monotonic()
    for record in cohort["runs"]:
        available = cohort["limits"]["queue_seconds"] - (time.monotonic() - clock)
        status = state["runs"][record["run_id"]]
        if available <= 0:
            status.update(status="not_started_deadline", ended_utc=stamp())
            continue
        attempt = ROOT / "runs" / record["run"] / "artifacts/attempt-01-spark"
        attempt.mkdir(parents=True, exist_ok=False)
        source = attempt / "source"
        source.mkdir()
        import shutil
        shutil.copy2(ROOT / "runs" / record["run"] / "config.json", source / "config.json")
        shutil.copytree(ROOT / "src/cifar_eos", source / "cifar_eos", ignore=shutil.ignore_patterns("__pycache__"))
        output = attempt / "production"
        command = [sys.executable, "-m", "src.cifar_eos.train", "--config", str(source / "config.json"),
                   "--inputs", cohort["remote_inputs"], "--output", str(output), "--optimizer", record["optimizer"]]
        assert file_hash(source / "config.json") == record["config_sha256"]
        state["current_run"] = record["run_id"]
        status.update(status="running", started_utc=stamp())
        write_json(HERE / "queue_state.json", state)
        with (attempt / "production.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timeout = min(available, cohort["limits"]["run_seconds"])
            try:
                code = child.wait(timeout=timeout)
                if code:
                    raise RuntimeError(f"Runner exited{code}; see production.log")
                manifest = json.loads((output / "manifest.json").read_text())
                assert manifest["status"] == "complete"
                assert manifest["completed_updates"] == 20000
                status.update(status="complete", ended_utc=stamp(), completed_updates=20000)
            except Exception as error:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                status.update(status="failed", ended_utc=stamp(), error=repr(error))
                mark_failed(output, repr(error))
        inventory(attempt)
        write_json(HERE / "queue_state.json", state)
    state.update(status="complete" if all(r["status"] == "complete" for r in state["runs"].values()) else "finished_with_failures",
                 completed_utc=stamp(), current_run=None, elapsed_seconds=time.monotonic() - clock)
    write_json(HERE / "queue_state.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
