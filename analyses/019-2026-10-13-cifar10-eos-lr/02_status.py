"""Read-only bounded queue/progress snapshot; no live log copies."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def last_json(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 32768))
        lines = source.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    queue = json.loads((HERE / "queue_state.json").read_text())
    result = {"utc": datetime.now(timezone.utc).isoformat(), "queue": queue, "runs": {}}
    completed = 0
    remaining = 0.0
    for record in cohort["runs"]:
        rid = record["run_id"]
        state = queue["runs"][rid]
        output = ROOT / "runs" / record["run"] / "artifacts/attempt-01-spark/production"
        summary = output / "summary.json"
        event = last_json(output / "events.jsonl")
        curvature = last_json(output / "curvature.jsonl")
        estimate = (6.16 if record["optimizer"] == "gd" else 8.43) * 60
        if state["status"] == "complete":
            info = json.loads(summary.read_text())
            completed += info["completed_updates"]
            result["runs"][rid] = {"status": "complete", "completed_updates": info["completed_updates"], "elapsed_seconds": info["elapsed_seconds"], "final_train": info["final_train"], "final_test": info["final_test"]}
        elif state["status"] == "running":
            result["runs"][rid] = {"status": "running", "progress": event, "curvature": curvature}
            if event:
                completed += event["completed_updates"]
                estimate = event["remaining_seconds_at_observed_rate"]
            remaining += estimate
        elif state["status"] == "queued":
            remaining += estimate
        else:
            result["runs"][rid] = {"status": state["status"], "progress": event}
    result.update(completed_updates=completed, requested_updates=160000, training_remaining_seconds_estimate=remaining)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
