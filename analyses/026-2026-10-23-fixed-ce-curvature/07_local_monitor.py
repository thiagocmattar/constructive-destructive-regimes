"""Read-only local progress watcher and a deadline guard for its own queue PID."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    lease = json.loads((HERE / "local_execution.json").read_text())
    assert lease["owner"] == "Analysis026-current-conversation"
    while True:
        status_path = ROOT / "worker-control/status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("output"):
                events = ROOT / status["output"] / "events.jsonl"
                if events.exists():
                    status["latest_events"] = [json.loads(x) for x in events.read_text().splitlines()[-2:]]
            status["observed_epoch"] = time.time()
            (HERE / "latest-status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(status) + "\n")
            if status["status"] in ("complete", "failed", "calibration_exceeds_envelope"):
                return
        if time.time() >= lease["hard_stop_epoch"]:
            # The PID was returned by this conversation's Popen; no unrelated process selection.
            try:
                os.kill(lease["queue_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
