"""Task-owned SSH transport, hash-verified retrieval and independent bill guard."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / "tools/bin/runpodctl.exe"


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def transport(lease, scp=False):
    return ["scp" if scp else "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(ROOT / "cache/runpod/known_hosts"),
            "-i", lease["ssh_key"], "-P" if scp else "-p", str(lease["port"])]


def remote(lease, program, timeout=45):
    result = subprocess.run(transport(lease) + [f"root@{lease['host']}", "bash -s"],
                            input=program.encode(), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace") + result.stdout.decode(errors="replace"))
    return result.stdout.decode()


def cli(*args, check=True):
    result = subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=45)
    if check and result.returncode:
        raise RuntimeError(result.stderr)
    return result


def assert_owned(lease):
    assert lease["owner"] == "Analysis026-current-conversation"
    pod = json.loads(cli("pod", "get", lease["pod_id"]).stdout)
    assert pod["id"] == lease["pod_id"] and pod["name"] == lease["pod_name"]
    return pod


def stop_owned(lease, reason):
    before = assert_owned(lease)
    result = cli("pod", "stop", lease["pod_id"], check=False)
    after = json.loads(cli("pod", "get", lease["pod_id"]).stdout)
    save(HERE / "compute_stop.json", {"utc_epoch": time.time(), "reason": reason,
         "pod_id": lease["pod_id"], "before": before.get("runtimeStatus"),
         "after": after.get("runtimeStatus"), "exit_code": result.returncode})


def collect(lease):
    # Each source file is addressed relative to the task root; preserve run ownership.
    script = """import hashlib,json
from pathlib import Path
root=Path('/workspace/h022')
files=[]
for base in sorted((root/'runs').glob('*-2026-10-23-fixed-ce-*/artifacts')):
 for p in sorted(base.rglob('*')):
  if p.is_file() and not p.name.endswith('.tmp'):
   with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
   files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest})
print(json.dumps({'files':files,'total_bytes':sum(f['bytes'] for f in files)}))
"""
    manifest = json.loads(remote(lease, "python3 - <<'H022_INVENTORY'\n" + script + "H022_INVENTORY\n", timeout=180))
    save(HERE / "retrieval_manifest.json", manifest)
    # One archive avoids thousands of SSH handshakes. Author the member list remotely.
    program = """cd /workspace/h022
python3 - <<'H022_ARCHIVE'
from pathlib import Path
import tarfile
with tarfile.open('/workspace/h022-results.tar','w') as archive:
 for base in sorted(Path('runs').glob('*-2026-10-23-fixed-ce-*/artifacts')):
  archive.add(base,arcname=str(base))
H022_ARCHIVE
"""
    remote(lease, program, timeout=180)
    archive = ROOT / "cache/fixed-ce-curvature/results.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(transport(lease, True) + ["-q", f"root@{lease['host']}:/workspace/h022-results.tar", str(archive)],
                            capture_output=True, text=True, timeout=1200)
    if result.returncode:
        raise RuntimeError(result.stderr)
    import tarfile
    with tarfile.open(archive) as source:
        source.extractall(ROOT, filter="data")
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"], item["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], item["path"]
    save(HERE / "retrieved.json", {"status": "verified", "utc_epoch": time.time(),
         "pod_id": lease["pod_id"], "files": len(manifest["files"]),
         "bytes": manifest["total_bytes"]})
    return manifest


def monitor(lease):
    assert_owned(lease)
    next_poll = 0
    while True:
        if time.time() >= lease["stop_deadline_epoch"]:
            stop_owned(lease, "four-hour independent bill guard")
            return
        if time.time() >= next_poll:
            try:
                text = remote(lease, "cat /workspace/h022/worker-control/status.json\n")
                status = json.loads(text)
                if status.get("output"):
                    progress = remote(lease, "tail -n 2 /workspace/h022/" + status["output"] + "/events.jsonl\n")
                    status["latest_events"] = [json.loads(line) for line in progress.splitlines()]
                status["observed_epoch"] = time.time()
                save(HERE / "latest-status.json", status)
                with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(status) + "\n")
                if status["status"] in ("complete", "failed", "calibration_exceeds_envelope"):
                    collect(lease)
                    stop_owned(lease, "terminal evidence retrieved and SHA256 verified")
                    return
            except Exception as error:
                save(HERE / "controller.json", {"utc_epoch": time.time(), "error": repr(error)})
            next_poll = time.time() + 600
        time.sleep(min(30, max(1, lease["stop_deadline_epoch"] - time.time())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("monitor", "collect", "stop", "status"))
    args = parser.parse_args()
    lease = json.loads((HERE / "lease.json").read_text())
    if args.mode == "monitor":
        monitor(lease)
    elif args.mode == "collect":
        collect(lease)
    elif args.mode == "stop":
        stop_owned(lease, "explicit task closeout")
    else:
        print(remote(lease, "cat /workspace/h022/worker-control/status.json\n"))


if __name__ == "__main__":
    main()
