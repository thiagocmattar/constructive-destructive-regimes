"""Independent lease guard; preserve network evidence when stopping owned compute."""
from pathlib import Path
import json
import subprocess
import time
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    if os.name == "nt":
        import ctypes
        # Keep this computer awake while its independent spending guard is active.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    owned = json.loads((HERE / "created-resources.json").read_text())
    lease = json.loads((HERE / "lease.json").read_text())
    assert owned["pod_id"] == lease["pod_id"]
    while time.time() < lease["deadline_epoch"]:
        time.sleep(min(30, max(0, lease["deadline_epoch"] - time.time())))
    controller = HERE / "local-controller.json"
    if controller.exists() and json.loads(controller.read_text()).get("status") == "retrieved_cleanup_requested":
        return
    results = []
    for pod in owned.get("pod_ids", [owned["pod_id"]]):
        result = subprocess.run([str(ROOT / "tools/bin/runpodctl.exe"), "pod", "stop", pod],
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        results.append({"pod_id": pod, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    (HERE / "deadline-receipt.json").write_text(json.dumps({"pod_ids": owned.get("pod_ids", [owned["pod_id"]]),
        "action": "stop", "deadline_epoch": lease["deadline_epoch"], "checked_epoch": time.time(),
        "results": results,
        "network_evidence_preserved": owned["volume_id"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
