"""Freeze a small input archive, stage it once and launch the detached queue."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.fixed_ce_curvature.core import save_json, sha, stamp
spec = importlib.util.spec_from_file_location("control", HERE / "05_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def package():
    inventory_path = HERE / "source_inventory.json"
    old = json.loads(inventory_path.read_text())
    paths = [ROOT / item["path"] for item in old["files"]]
    paths += [HERE / "04_queue.py", HERE / "03_check_implementation.py", HERE / "cohort.json"]
    paths = sorted(set(paths))
    source = {"utc": stamp(), "files": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
               "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]}
    save_json(inventory_path, source)
    archive = ROOT / "cache/fixed-ce-curvature/input.tar"
    with tarfile.open(archive, "w") as tar:
        for path in paths + [inventory_path]:
            tar.add(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    save_json(HERE / "source_packet.json", {"bytes": archive.stat().st_size, "sha256": sha(archive),
                                          "utc": stamp(), "path": str(archive)})
    print(json.dumps({"bytes": archive.stat().st_size, "sha256": sha(archive)}))


def stage():
    lease = json.loads((HERE / "lease.json").read_text())
    packet = json.loads((HERE / "source_packet.json").read_text())
    archive = Path(packet["path"])
    result = subprocess.run(control.transport(lease, True) + ["-q", str(archive),
                            f"root@{lease['host']}:/workspace/h022-input.tar"],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    program = f"""set -euo pipefail
mkdir -p /workspace/h022/worker-control
cd /workspace/h022
test "$(sha256sum /workspace/h022-input.tar | cut -d ' ' -f 1)" = '{packet['sha256']}'
tar -xf /workspace/h022-input.tar
python3 - <<'H022_IMPORTS'
import torch
print('TEMPLATE_TORCH',torch.__version__)
H022_IMPORTS
python3 -m pip install --disable-pip-version-check scipy
python3 analyses/026-2026-10-23-fixed-ce-curvature/03_check_implementation.py
python3 -m pip freeze > worker-control/environment-lock.txt
test ! -e worker-control/queue.lock
setsid env CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u analyses/026-2026-10-23-fixed-ce-curvature/04_queue.py --deadline-epoch {lease['training_deadline_epoch']} --created-epoch {lease['created_epoch']} > worker-control/queue.log 2>&1 < /dev/null &
queue_pid=$!
echo "$queue_pid" > worker-control/queue.pid
setsid python3 - "$queue_pid" {lease['training_deadline_epoch']} <<'H022_WATCHDOG' > worker-control/watchdog.log 2>&1 < /dev/null &
import os,signal,sys,time
pid=int(sys.argv[1]); deadline=float(sys.argv[2])
while time.time()<deadline:
 time.sleep(min(30,max(0.1,deadline-time.time())))
try:os.killpg(pid,signal.SIGTERM)
except ProcessLookupError:pass
H022_WATCHDOG
echo "QUEUE_PID=$queue_pid"
"""
    print(control.remote(lease, program, timeout=240), flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log = (ROOT / "cache/fixed-ce-curvature/controller.log").open("ab")
    process = subprocess.Popen([sys.executable, str(HERE / "05_control.py"), "monitor"],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                               creationflags=flags)
    save_json(HERE / "controller.json", {"pid": process.pid, "utc": stamp(), "status": "monitoring"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("package", "stage"))
    mode = parser.parse_args().mode
    package() if mode == "package" else stage()
