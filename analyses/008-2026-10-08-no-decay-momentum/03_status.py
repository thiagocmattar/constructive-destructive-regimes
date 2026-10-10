"""One bounded read-only cohort snapshot, including timing and warning fields."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
cohort = json.loads((HERE / "cohort.json").read_text())
state_path = HERE / "queue_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "not_launched"}
snapshot = {"timestamp": datetime.now(timezone.utc).isoformat(), "queue": state, "runs": {}}
for record in cohort["runs"]:
    folder = ROOT / "runs" / record["run"] / "artifacts/attempt-01/full-pass"
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text())
        latest = None
        if (folder / "events.jsonl").exists():
            with (folder / "events.jsonl").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                lines = stream.read().split(b"\n")[:-1]
                latest = next((e for line in reversed(lines) if line.startswith(b"{")
                               and (e := json.loads(line)).get("event") == "update"), None)
        snapshot["runs"][record["run_id"]] = {"manifest": manifest, "latest_update": latest}
if state.get("current_run") in snapshot["runs"]:
    item = snapshot["runs"][state["current_run"]]
    event = item["latest_update"]
    if event:
        remaining = event["remaining_training_seconds"]
        queued = sum(v["status"] == "queued" for v in state["runs"].values())
        snapshot["estimated_cohort_remaining_hours"] = remaining / 3600 + queued * 5.1
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(item["manifest"]["latest_event_at"])).total_seconds()
        snapshot["warning_stale_progress"] = age > 180
        snapshot["warning_overflow_retries"] = event["overflow_retries"] > 0
snapshot["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(json.dumps(snapshot, allow_nan=False))
