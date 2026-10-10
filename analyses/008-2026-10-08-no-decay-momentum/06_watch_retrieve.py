"""Bounded local monitoring, verified retrieval and automatic figure generation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOST = "tcml-spark@spark-2277.local"
REMOTE = "experiments/constructive-destructive-regimes/no-decay-2026-10-08"
PYTHON_REMOTE = "../spark-calibration-2026-10-08/venv/bin/python"
MARKER = Path("C:/Users/thima/.codex/plugins/cache/openai-primary-runtime/pdf/26.909.12148/skills/pdf/container_tools/mark_artifact_operation_started.mjs")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST]
SCP = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def remote(command, timeout=60):
    return subprocess.run(SSH + [f"cd {REMOTE} && {command}"], capture_output=True,
                          text=True, timeout=timeout, check=True).stdout


def mark_once():
    receipt = HERE / "artifact-operation-started.json"
    if not receipt.exists():
        subprocess.run(["node", str(MARKER), "--operation-kind", "create",
                        "--expected-output-count", "7", "--output-format", "pdf"], check=True)
        save(receipt, {"operation": "create", "expected_final_pdfs": 7})


def render(destination):
    for path in (destination / "figures").glob("*.pdf"):
        with fitz.open(path) as document:
            if len(document) != 1:
                raise ValueError(f"Unexpected PDF page count: {path}")
            page = document[0]
            if not page.get_text().strip():
                raise ValueError(f"Empty plot text: {path}")
            page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(destination / "scratch" / (path.stem + "-render.png"))


def retrieve(record, status):
    run = ROOT / "runs" / record["run"]
    prelaunch = run / "prelaunch"
    receipt = prelaunch / "retrieval-receipt.json"
    if receipt.exists() and json.loads(receipt.read_text())["accepted"]:
        if not (prelaunch / "figure-generation.json").exists():
            mark_once()
            from importlib import import_module
            module = import_module("04_plot")
            module.plot_one(run)
            render(run)
            save(prelaunch / "figure-generation.json", {"generated": True, "renders_present": True,
                                                       "visual_review": "pending_model_inspection"})
        return
    relative = f"runs/{record['run']}/artifacts/attempt-01/full-pass"
    if status == "failed":
        remote(f"{PYTHON_REMOTE} analyses/{HERE.name}/05_remote_inventory.py {relative}", timeout=300)
    inventory_sha = remote(f"sha256sum {relative}/transfer_inventory.json").split()[0]
    parent = run / "artifacts/attempt-01"
    parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(SCP + ["-r", f"{HOST}:{REMOTE}/{relative}", str(parent)], check=True, timeout=1800)
    folder = parent / "full-pass"
    inventory_path = folder / "transfer_inventory.json"
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == inventory_sha
    inventory = json.loads(inventory_path.read_text())
    for item in inventory["files"]:
        path = (folder / item["path"]).resolve()
        assert path.is_relative_to(folder.resolve()) and not path.is_symlink()
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    if status == "completed":
        assert json.loads((folder / "verification.json").read_text())["verified"]
    save(receipt, {"accepted": True, "files_verified": len(inventory["files"]),
                   "inventory_sha256": inventory_sha, "bytes": sum(i["bytes"] for i in inventory["files"]),
                   "scope": "full_pass" if status == "completed" else "failed_attempt_bytes_only",
                   "remote_state_retained": True})
    mark_once()
    from importlib import import_module
    module = import_module("04_plot")
    module.plot_one(run)
    render(run)
    save(prelaunch / "figure-generation.json", {"generated": True, "renders_present": True,
                                               "visual_review": "pending_model_inspection"})


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    deadline = time.time() + 26 * 3600
    completed = set()
    state = {"status": "monitoring", "started_at_epoch": time.time(), "retrieved_runs": []}
    save(HERE / "local-controller.json", state)
    while time.time() < deadline:
        if (HERE / "STOP_MONITORING").exists():
            state["status"] = "stopped_by_user_marker"
            save(HERE / "local-controller.json", state)
            return
        try:
            snapshot = json.loads(remote(f"{PYTHON_REMOTE} analyses/{HERE.name}/03_status.py"))
            save(HERE / "latest-status.json", snapshot)
            with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
            print(json.dumps({"timestamp": snapshot["timestamp"], "status": snapshot["queue"]["status"],
                              "current_run": snapshot["queue"].get("current_run"),
                              "remaining_hours": snapshot.get("estimated_cohort_remaining_hours"),
                              "gpu": snapshot["gpu"]}), flush=True)
            for record in cohort["runs"]:
                condition = snapshot["queue"].get("runs", {}).get(record["run_id"], {}).get("status")
                if condition in ("completed", "failed") and record["run_id"] not in completed:
                    retrieve(record, condition)
                    completed.add(record["run_id"])
                    state["retrieved_runs"] = sorted(completed)
                    save(HERE / "local-controller.json", state)
            if snapshot["queue"]["status"] in ("completed", "completed_with_failures") and len(completed) == 3:
                mark_once()
                from importlib import import_module
                module = import_module("04_plot")
                module.plot_joint()
                render(HERE)
                state.update(status="completed_pending_visual_review", finished_at_epoch=time.time())
                save(HERE / "local-controller.json", state)
                return
        except Exception:
            state["last_error"] = traceback.format_exc()
            save(HERE / "local-controller.json", state)
            print(state["last_error"], flush=True)
        time.sleep(60)
    state["status"] = "monitoring_deadline_reached_remote_state_preserved"
    save(HERE / "local-controller.json", state)


if __name__ == "__main__":
    main()
