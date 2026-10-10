"""Native SSH actions for the single owned Run002 worker; no credentials in logs."""
from pathlib import Path
import json
import shlex
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
RUN_DIR = "/workspace/constructive/runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry"
STATUS = f"python3 {RUN_DIR}/prelaunch/remote_status.py"
PYTHON = "/opt/baseline-env/bin/python"
COMMANDS = {
    "bootstrap02": f"set -e; mkdir -p /workspace/constructive; tar -xzf /workspace/run002-source-attempt02.tar.gz -C /workspace/constructive; mv /workspace/constructive/input-cache /workspace/minipile-cache; cp /workspace/lease.json {RUN_DIR}/prelaunch/lease.json; cp {RUN_DIR}/prelaunch/source-inventory-attempt2.json {RUN_DIR}/prelaunch/executed-source-inventory.json; cd {RUN_DIR}; setsid bash 00_setup_remote.sh > /workspace/constructive/setup.log 2>&1 < /dev/null & echo $!",
    "accept_cache02": f"setsid python3 {RUN_DIR}/prelaunch/accept_local_cache.py > /workspace/constructive/transfer.log 2>&1 < /dev/null & echo $!",
    "launch_archive02": f"setsid bash {RUN_DIR}/prelaunch/archive_attempt02.sh > /workspace/constructive/archive.log 2>&1 < /dev/null & echo $!",
    "archive_status02": "tail -n 4 /workspace/constructive/archive.log; cat /opt/run002-attempt-02.tar.gz.sha256 /opt/run002-attempt-02.tar.gz.bytes",
    "launch_calibration02": f"cd {RUN_DIR}; setsid bash prelaunch/run_attempt02.sh calibrate > /workspace/constructive/calibration.log 2>&1 < /dev/null & echo $!",
    "launch_full02": f"cd {RUN_DIR}; setsid bash prelaunch/run_attempt02.sh full-pass > /workspace/constructive/full-pass.log 2>&1 < /dev/null & echo $!",
    "setup_status": "tail -n 4 /workspace/constructive/setup.log; cat /workspace/constructive/transfer.log; ps -eo pid,etime,args",
    "calibration_status": STATUS + " calibration",
    "calibration_local_status": STATUS + " calibration-local",
    "calibration_thread_status": STATUS + " calibration-thread",
    "full_status": STATUS + " full-pass",
    "audit_gpu": f"cd /workspace/constructive; {PYTHON} {RUN_DIR}/prelaunch/audit_gpu_adam.py",
    "launch_calibration": f"cd {RUN_DIR}; setsid bash 02_train_remote.sh calibrate > /workspace/constructive/calibration.log 2>&1 < /dev/null & echo $!",
    "launch_full": f"cd {RUN_DIR}; setsid bash 02_train_remote.sh full-pass > /workspace/constructive/full-pass.log 2>&1 < /dev/null & echo $!",
    "launch_staging": f"setsid bash {RUN_DIR}/prelaunch/stage_cache.sh > /workspace/constructive/staging.log 2>&1 < /dev/null & echo $!",
    "staging_status": "cat /workspace/constructive/staging.log; ls -l /opt/minipile-cache/train/tokens.int32.bin",
    "launch_calibration_local": f"cd {RUN_DIR}; setsid bash prelaunch/run_staged.sh calibrate > /workspace/constructive/calibration-local.log 2>&1 < /dev/null & echo $!",
    "launch_full_local": f"cd {RUN_DIR}; setsid bash prelaunch/run_staged.sh full-pass > /workspace/constructive/full-pass.log 2>&1 < /dev/null & echo $!",
    "launch_calibration_thread": f"cd {RUN_DIR}; setsid timeout 600 {PYTHON} prelaunch/train_with_cpu_threads.py --cache /opt/minipile-cache --output artifacts/attempt-01/calibration-thread --calibrate-updates 3 > /workspace/constructive/calibration-thread.log 2>&1 < /dev/null & echo $!",
    "hardware": "nvidia-smi --query-gpu=utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit,pstate --format=csv,noheader; cat /sys/fs/cgroup/cpu.max; cat /sys/fs/cgroup/cpu.stat; lscpu | head -n 18",
    "launch_archive01": f"setsid bash {RUN_DIR}/prelaunch/archive_attempt01.sh > /workspace/constructive/archive.log 2>&1 < /dev/null & echo $!",
    "archive_status": "tail -n 4 /workspace/constructive/archive.log; cat /workspace/run002-attempt-01.tar.gz.sha256",
    "logs": "tail -n 8 /workspace/constructive/full-pass.log; tail -n 8 /workspace/constructive/calibration.log",
    "memory": "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; df -h /workspace /opt",
}
action = sys.argv[1]
transport = json.loads((HERE / "transport.json").read_text())
if action == "verify_source":
    code = "from pathlib import Path; import hashlib,json; root=Path('/workspace/constructive'); inv=json.loads((root/'runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/prelaunch/executed-source-inventory.json').read_text()); "
    code += "assert all(hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in inv['files']); print('ALL_SOURCE_HASHES_MATCH')"
    command = "python3 -c " + shlex.quote(code)
else:
    command = COMMANDS[action]
argv = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "-o",
        "UserKnownHostsFile=" + str(WORKSPACE / "cache/runpod/known_hosts"),
        "-i", "C:/Users/thima/.ssh/id_ed25519_runpod", "-p", str(transport["port"]),
        transport["user"] + "@" + transport["host"], command]
result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
if action in ("calibration_status", "calibration_local_status", "calibration_thread_status", "full_status") and result.returncode == 0:
    snapshot = json.loads(result.stdout)
    snapshot["controller_pod_id"] = transport["pod_id"]
    with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot) + "\n")
sys.exit(result.returncode)
