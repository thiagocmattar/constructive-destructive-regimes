"""Detached sequential training, verification and remote figure generation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERIFY = ROOT / "analyses/008-2026-10-08-no-decay-momentum/02_verify.py"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def inventory(folder):
    entries = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "transfer_inventory.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entries.append(dict(path=path.relative_to(folder).as_posix(), bytes=path.stat().st_size, sha256=digest))
    save(folder / "transfer_inventory.json", dict(files=entries))


def bounded(command, log, seconds):
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise RuntimeError(f"Process-group deadline exceeded ({seconds}s)")
        if code:
            raise RuntimeError(f"Command exited {code}; see {log.relative_to(ROOT)}")


def main():
    for item in json.loads((HERE / "source_inventory.json").read_text())["files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    cohort = json.loads((HERE / "cohort.json").read_text())
    state_path = HERE / "queue_state.json"
    if state_path.exists():
        raise RuntimeError("An executed queue exists; preserve this attempt.")
    state = dict(status="running", started_at=now(), current_run=None, plot_status="not_started",
                 runs={r["run_id"]:dict(status="queued", peak_lr=r["peak_lr"]) for r in cohort["runs"]})
    save(state_path, state)
    for record in cohort["runs"]:
        rid = record["run_id"]
        run = ROOT / "runs" / record["run"]
        attempt = run / "artifacts/attempt-01"
        attempt.mkdir(parents=True, exist_ok=False)
        current_output = None
        state["current_run"] = rid
        try:
            for label, timeout in (("calibration", 900), ("full-pass", 8*3600)):
                current_output = attempt / label
                state["runs"][rid].update(status="calibrating" if label == "calibration" else "running", phase=label, started_at=now())
                save(state_path, state)
                command = [sys.executable, str(run / "01_train.py"), "--config", str(run / "config.json"),
                           "--cache", str(ROOT / "data"), "--output", str(current_output)]
                if label == "calibration":
                    command += ["--calibrate-updates", "3"]
                bounded(command, attempt / (label+".log"), timeout)
                bounded([sys.executable, str(VERIFY), str(current_output)], attempt / (label+"-verification.log"), 600)
                assert json.loads((current_output / "verification.json").read_text())["verified"]
            state["runs"][rid].update(status="completed", finished_at=now())
        except Exception as error:
            state["runs"][rid].update(status="failed", finished_at=now(), error=str(error))
            if current_output and (current_output / "manifest.json").exists():
                path = current_output / "manifest.json"
                manifest = json.loads(path.read_text())
                if manifest["status"] != "completed":
                    manifest.update(status="failed", finished_at=now(), error=str(error))
                    save(path, manifest)
                if not (current_output / "transfer_inventory.json").exists():
                    inventory(current_output)
            print(traceback.format_exc(), flush=True)
        save(state_path, state)
    state.update(current_run=None, status="completed" if all(v["status"] == "completed" for v in state["runs"].values()) else "completed_with_failures",
                 finished_at=now(), plot_status="running")
    save(state_path, state)
    plot_python = ROOT / "plot-venv/bin/python"
    try:
        bounded([str(plot_python), str(HERE / "03_plot.py")], HERE / "plot-generation.log", 600)
        bounded([str(plot_python), str(HERE / "04_record_results.py")], HERE / "record-results.log", 120)
        state.update(plot_status="completed", artifacts_ready_at=now())
    except Exception as error:
        state.update(plot_status="failed", plot_error=str(error))
        print(traceback.format_exc(), flush=True)
    save(state_path, state)
    payload = [HERE / name for name in ("README.md", "queue_state.json", "comparison_data.json", "summary.csv", "RESULTS.md", "figure_verification.json") if (HERE / name).exists()]
    payload += sorted((HERE / "figures").glob("*.pdf")) + sorted((HERE / "observations").glob("*.md"))
    for record in cohort["runs"]:
        run = ROOT / "runs" / record["run"]
        payload += [run / "README.md"] + sorted((run / "observations").glob("*.md"))
    entries = []
    for path in payload:
        entries.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(HERE / "result_inventory.json", dict(files=entries, finished_at=now()))
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
