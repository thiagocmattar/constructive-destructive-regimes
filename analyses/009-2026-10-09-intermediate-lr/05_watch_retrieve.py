"""Bounded read-only monitoring, verified retrieval and owned-resource cleanup."""
from pathlib import Path
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("operate009", HERE / "04_operate.py")
operate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operate)
bundle_spec = importlib.util.spec_from_file_location("bundles009", HERE / "08_retrieve_bundles.py")
bundles = importlib.util.module_from_spec(bundle_spec)
bundle_spec.loader.exec_module(bundles)
CLI = ROOT / "tools/bin/runpodctl.exe"


def api(*args):
    result = subprocess.run([str(CLI), *args], capture_output=True, text=True, encoding="utf-8", timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    owned = json.loads((HERE / "created-resources.json").read_text())
    lease = json.loads((HERE / "lease.json").read_text())
    state = {"status": "monitoring", "pod_ids": owned.get("pod_ids", [owned["pod_id"]]),
             "retrieved_runs": [], "pid": __import__("os").getpid(), "started_epoch": time.time()}
    operate.save(HERE / "local-controller.json", state)
    while time.time() < lease["deadline_epoch"]:
        try:
            snapshot = operate.snapshot()
            operate.save(HERE / "latest-status.json", snapshot)
            with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot) + "\n")
            shown = []
            for record, row in zip(operate.records(), snapshot["runs"]):
                data = row["full-pass"]
                recent = data.get("recent_updates", [])
                latest = recent[-1] if recent else {}
                warnings = []
                if data.get("wrapper", {}).get("status") == "running" and data.get("event_age_seconds", 0) > 300:
                    warnings.append("no_training_event_for_300_seconds")
                if latest and not math.isfinite(latest["loss"]):
                    warnings.append("nonfinite_loss")
                if any(e.get("overflow_retries", 0) for e in recent):
                    warnings.append("overflow_retry_in_recent_updates")
                mean_seconds = statistics.mean(e["update_seconds"] for e in recent) if recent else None
                shown.append({"run_id": row["run_id"], "updates": latest.get("update"),
                              "loss": latest.get("loss"), "tokens_per_second":
                              latest.get("tokens_per_second"), "state": data.get("wrapper", {}).get("status"),
                              "remaining_training_seconds": (712-latest["update"])*mean_seconds if recent else None,
                              "warnings": warnings})
                if data.get("wrapper", {}).get("status") == "completed" and record["run_id"] not in state["retrieved_runs"]:
                    bundles.retrieve(record)
                    operate.retrieve_calibrations(record)
                    state["retrieved_runs"].append(record["run_id"])
                    operate.save(HERE / "local-controller.json", state)
                elif data.get("wrapper", {}).get("status") == "failed":
                    state["failed_run"] = row["run_id"]
            fleet_warnings = []
            if snapshot["network_used_bytes"] > 55_000_000_000:
                fleet_warnings.append("network_volume_near_60GB_quota")
            for key in ("gpu", "gpu_secondary"):
                for line in snapshot.get(key, "").splitlines():
                    if int(line.split(",")[2].strip()) >= 85:
                        fleet_warnings.append(key + "_temperature_at_least_85C")
            print(json.dumps({"epoch": snapshot["epoch"], "runs": shown, "gpu": snapshot["gpu"],
                              "gpu_secondary": snapshot.get("gpu_secondary"),
                              "network_used_bytes": snapshot["network_used_bytes"],
                              "fleet_warnings": fleet_warnings}), flush=True)
            if len(state["retrieved_runs"]) == 4:
                # Retrieval checks require scientific verification and every byte/hash, per run.
                for record in operate.records():
                    assert json.loads((ROOT / "runs" / record["run"] / "prelaunch/retrieval-receipt.json").read_text())["accepted"]
                operate.retrieve_execution_records()
                state["pod_delete_results"] = {pod: api("pod", "remove", pod) for pod in owned.get("pod_ids", [owned["pod_id"]])}
                state["volume_delete_result"] = api("network-volume", "delete", owned["volume_id"])
                state.update(status="retrieved_cleanup_requested", finished_epoch=time.time(),
                             verification="all_four_file_sizes_and_sha256_passed",
                             cleanup_readback_required=True)
                operate.save(HERE / "local-controller.json", state)
                return
        except Exception:
            state["last_error"] = traceback.format_exc()
            operate.save(HERE / "local-controller.json", state)
            print(state["last_error"], flush=True)
        time.sleep(min(60, max(0, lease["deadline_epoch"] - time.time())))
    # Preserve the independent network volume at the spending deadline.
    state["stop_results"] = {pod: api("pod", "stop", pod) for pod in owned.get("pod_ids", [owned["pod_id"]])}
    state.update(status="deadline_stop_requested_evidence_retained", finished_epoch=time.time(),
                 retained_volume=owned["volume_id"], retained_storage_gb=60,
                 storage_usd_per_month=4.2, cleanup_readback_required=True)
    operate.save(HERE / "local-controller.json", state)


if __name__ == "__main__":
    main()
