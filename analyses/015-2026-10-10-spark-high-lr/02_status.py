"""Read-only full-pass progress, warnings, memory and qualified completion ETC."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
state_path = HERE / "queue_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="not_launched")
snapshot = dict(timestamp=datetime.now(timezone.utc).isoformat(), queue=state, runs={})
for record in json.loads((HERE / "cohort.json").read_text())["runs"]:
    item = state.get("runs", {}).get(record["run_id"], {})
    phase = item.get("phase", "full-pass")
    folder = ROOT / "runs" / record["run"] / "artifacts/attempt-01" / phase
    latest = None
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text())
        if (folder / "events.jsonl").exists():
            with (folder / "events.jsonl").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-65536))
                for line in reversed(stream.read().split(b"\n")[:-1]):
                    if line.startswith(b"{"):
                        event = json.loads(line)
                        if event.get("event") == "update":
                            latest = event
                            break
        snapshot["runs"][record["run_id"]] = dict(phase=phase, manifest=manifest, latest_update=latest)
        if record["run_id"] == state.get("current_run") and latest:
            remaining = latest["remaining_training_seconds"]
            if phase == "calibration":
                remaining += 5.1*3600
            remaining += sum(v["status"] == "queued" for v in state["runs"].values())*5.1*3600
            snapshot["estimated_cohort_remaining_hours"] = remaining/3600
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(manifest["latest_event_at"])).total_seconds()
            snapshot["warning_stale_progress"] = item["status"] in ("running", "calibrating") and age > 180
            snapshot["warning_overflow_retries"] = latest["overflow_retries"] > 0
snapshot["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
print(json.dumps(snapshot, allow_nan=False))
