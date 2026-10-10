"""Bounded monitoring and parallel retrieval; each resource still needs MCP teardown."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("operate",HERE/"02_operate.py")
operate=importlib.util.module_from_spec(spec);spec.loader.exec_module(operate)
sys.stdout.reconfigure(encoding="utf-8")


def collect(row):
    run=ROOT/"runs"/row["run"]
    for script in [ROOT/"tools/download_scale_archive.py",HERE/"04_accept.py"]:
        result=subprocess.run([sys.executable,str(script),row["run"]],capture_output=True,text=True,encoding="utf-8",timeout=1500)
        (run/"prelaunch"/(script.stem+"-log.txt")).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-600:]+result.stdout[-600:])
    print(json.dumps({"run":row["run"][:3],"stage":"accepted; ready for owned-worker deletion"}),flush=True)


def status(row):
    run=ROOT/"runs"/row["run"]
    pre=run/"prelaunch"
    if (pre/"attempt1-retrieval.json").exists():
        return {"run":row["run"][:3],"stage":"accepted"}
    snapshot=json.loads(operate.native(row,"status"))
    manifest,event=snapshot.get("manifest",{}),snapshot.get("latest_update",{})
    result={"run":row["run"][:3],"stage":manifest.get("status"),"update":manifest.get("completed_updates"),
        "input_tokens":manifest.get("completed_input_tokens"),"loss":manifest.get("latest_loss"),
        "tokens_per_second":event.get("tokens_per_second"),"remaining_seconds":manifest.get("remaining_training_seconds"),
        "peak_memory_gb":event.get("peak_cuda_allocated_bytes",0)/1e9,"overflows":event.get("overflow_retries")}
    if manifest.get("status")=="failed":
        raise RuntimeError(row["run"]+": "+manifest.get("error","training failed"))
    if snapshot.get("verification",{}).get("verified"):
        receipt=pre/"archive-start.json"
        if not receipt.exists():
            result["archive_pid"]=operate.native(row,"archive").strip()
            receipt.write_text(json.dumps({"pod_id":json.loads((pre/"transport.json").read_text())["pod_id"],
                "started_at":datetime.now(timezone.utc).isoformat(),"pid":result["archive_pid"]},indent=2)+"\n")
        archive=operate.native(row,"archive-status")
        (pre/"archive-status.txt").write_text(archive)
        result["stage"]="archive-ready" if "ARCHIVE_READY" in archive else "archiving"
    return result


def main():
    rows=json.loads((HERE/"cohort.json").read_text())
    pending={}
    with ThreadPoolExecutor(max_workers=6) as monitors,ThreadPoolExecutor(max_workers=2) as transfers:
        while True:
            for run,future in list(pending.items()):
                if future.done():
                    future.result()
                    del pending[run]
            snapshots=list(monitors.map(status,rows))
            now=datetime.now(timezone.utc).isoformat()
            with (HERE/"monitoring.jsonl").open("a") as stream:
                stream.write(json.dumps({"at":now,"workers":snapshots})+"\n")
            print(json.dumps({"at":now,"workers":snapshots}),flush=True)
            for row,snapshot in zip(rows,snapshots):
                if snapshot["stage"]=="archive-ready" and row["run"] not in pending:
                    pending[row["run"]]=transfers.submit(collect,row)
            if all(s["stage"]=="accepted" for s in snapshots):
                break
            time.sleep(60)


if __name__=="__main__":main()
