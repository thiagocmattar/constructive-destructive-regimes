"""Bounded local retrieval controller; accepts each immutable completed run once."""
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent
HOST = "tcml-spark@spark-2277.local"


def command(arguments, timeout=120):
    result = subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=timeout)
    return result.stdout


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    remote = cohort["remote_root"]
    stage = ROOT / "cache/cifar10-eos-lr-results"
    stage.mkdir(exist_ok=True)
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    accepted_path = HERE / "retrieval.json"
    accepted = json.loads(accepted_path.read_text()) if accepted_path.exists() else {"status": "running", "started_utc": stamp(), "runs": {}}
    clock = time.monotonic()
    previous = None
    while time.monotonic() - clock < 4*3600 + 900:
        status = json.loads(command(["ssh", "-o", "BatchMode=yes", HOST, cohort["remote_python"], remote + "/analyses/019-2026-10-13-cifar10-eos-lr/02_status.py"], 30))
        write_json(scratch / "latest-status.json", status)
        with (HERE / "monitoring.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(status, allow_nan=False) + "\n")
        key = (status["queue"]["current_run"], status["completed_updates"] // 5000, status["queue"]["status"])
        if key != previous:
            print(json.dumps({"utc": status["utc"], "current_run": key[0], "completed_updates": status["completed_updates"],
                              "training_remaining_minutes": status["training_remaining_seconds_estimate"] / 60,
                              "status": key[2]}), flush=True)
            previous = key
        for record in cohort["runs"]:
            rid = record["run_id"]
            if status["queue"]["runs"][rid]["status"] != "complete" or rid in accepted["runs"]:
                continue
            relative = "runs/" + record["run"] + "/artifacts/attempt-01-spark"
            remote_archive = remote + "/transfer/Run" + rid + ".tar.gz"
            shell = ("mkdir -p " + shlex.quote(remote + "/transfer") + " && tar -czf " + shlex.quote(remote_archive)
                     + " -C " + shlex.quote(remote) + " " + shlex.quote(relative) + " && sha256sum " + shlex.quote(remote_archive))
            digest = command(["ssh", "-o", "BatchMode=yes", HOST, shell], 180).split()[0]
            archive = stage / ("Run" + rid + ".tar.gz")
            command(["scp", "-o", "BatchMode=yes", HOST + ":" + remote_archive, str(archive)], 180)
            assert file_hash(archive) == digest
            destination = ROOT / relative
            if destination.exists():
                raise FileExistsError(f"Existing immutable attempt: {destination}")
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    assert member.name == relative or member.name.startswith(relative + "/")
                    assert not member.issym() and not member.islnk()
                    assert (ROOT / member.name).resolve().is_relative_to(destination.resolve())
                bundle.extractall(ROOT, filter="data")
            inventory = json.loads((destination / "transfer_inventory.json").read_text())
            for item in inventory["files"]:
                path = destination / item["path"]
                assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"]
            accepted["runs"][rid] = {"utc": stamp(), "archive_bytes": archive.stat().st_size, "archive_sha256": digest,
                                      "files_verified": len(inventory["files"]), "verified_bytes": sum(f["bytes"] for f in inventory["files"])}
            write_json(accepted_path, accepted)
            print(json.dumps({"retrieved": rid, **accepted["runs"][rid]}), flush=True)
        if status["queue"]["status"] != "running":
            write_json(HERE / "queue_state.json", status["queue"])
            accepted.update(status="complete" if len(accepted["runs"]) == 8 else "finished_with_failures", ended_utc=stamp())
            write_json(accepted_path, accepted)
            if accepted["status"] != "complete":
                raise RuntimeError("Queue finished with missing/failed runs; preserve partial evidence.")
            print("All eight production runs retrieved and hash-verified.", flush=True)
            return
        time.sleep(20)
    raise TimeoutError("Retrieval controller deadline reached.")


if __name__ == "__main__":
    main()
