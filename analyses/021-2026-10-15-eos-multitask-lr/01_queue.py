"""Execute frozen production inputs sequentially with persistent bounded status."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json

HERE=Path(__file__).parent


def main():
    cohort=json.loads((HERE/"cohort.json").read_text())
    # Wait without allocating a GPU or altering the predecessor's inputs.
    statusfile=HERE/"queue_state.json"
    assert not statusfile.exists(), "Queue already launched."
    waiting=time.monotonic()
    write_json(statusfile,{"status":"waiting_for_predecessor","started_utc":stamp(),"runs":{}})
    while True:
        upstream=json.loads(Path(cohort["predecessor_state"]).read_text())
        if upstream["status"] in ("complete","finished_with_failures"):
            break
        if time.monotonic()-waiting>cohort["wait_max_seconds"]:
            write_json(statusfile,{"status":"predecessor_wait_deadline","runs":{}})
            raise TimeoutError("Predecessor has not reached a terminal state; GPU left alone.")
        time.sleep(300)
    # Starting science after the wait preserves a full bounded execution window.
    statusfile.unlink()
    for item in json.loads((HERE/"source_inventory.json").read_text())["files"]:
        p=ROOT/item["path"]
        assert p.stat().st_size==item["bytes"] and file_hash(p)==item["sha256"],item["path"]
    statusfile=HERE/"queue_state.json"
    if statusfile.exists():raise FileExistsError("Queue already executed; preserve it.")
    state={"status":"running","started_utc":stamp(),"current_run":None,"runs":{}}
    write_json(statusfile,state)
    begin=time.monotonic()
    for run in cohort["runs"]:
        if run["kind"]=="reused_gd_prefix":
            state["runs"][run["run_id"]]={"status":"reused_verified_source","updates":run["updates"]}
            continue
        remaining=cohort["hard_queue_seconds"]-(time.monotonic()-begin)
        if remaining<=0:
            state["runs"][run["run_id"]]={"status":"not_started_deadline"}
            write_json(statusfile,state)
            continue
        attempt=ROOT/"runs"/run["run"] / "artifacts/attempt-01-spark"
        attempt.mkdir(parents=True,exist_ok=False)
        source=attempt/"source"
        source.mkdir()
        for name in ("eos_tasks","cifar_eos"):
            shutil.copytree(ROOT/"src"/name,source/"src"/name,ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT/"runs"/run["run"]/"config.json",source/"config.json")
        assert file_hash(source/"config.json")==run["config_sha256"]
        command=[sys.executable,"-m","src.eos_tasks.train","--config",str(source/"config.json"),
            "--cache",cohort["remote_cache"],"--mlp-cache",cohort["remote_mlp_cache"],"--output",str(attempt/"production")]
        env=dict(os.environ,PYTHONPATH=str(source)+":"+cohort["remote_dependencies"])
        state["current_run"]=run["run_id"]
        record={"status":"running","started_utc":stamp(),"task":run["task"],"method":run["method"],"seed":run["seed"]}
        state["runs"][run["run_id"]]=record
        write_json(statusfile,state)
        with (attempt/"production.log").open("x") as log:
            child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record["pid"]=child.pid
            write_json(statusfile,state)
            try:
                code=child.wait(timeout=min(remaining,run["maximum_seconds"]))
                if code:raise RuntimeError(f"Runner exit{code}; see persistent log.")
                manifest=json.loads((attempt/"production/manifest.json").read_text())
                assert manifest["status"]=="complete" and manifest["completed_updates"]==run["updates"]
                record.update(status="complete",completed_updates=run["updates"])
            except Exception as error:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                record.update(status="failed_or_deadline",error=repr(error))
                manifestfile=attempt/"production/manifest.json"
                if manifestfile.exists():
                    manifest=json.loads(manifestfile.read_text());manifest.update(status="failed_or_deadline",error=repr(error))
                    write_json(manifestfile,manifest)
            record["ended_utc"]=stamp()
        write_json(attempt/"transfer_inventory.json",{"files":[{"path":p.relative_to(attempt).as_posix(),
            "bytes":p.stat().st_size,"sha256":file_hash(p)} for p in sorted(attempt.rglob("*"))
            if p.is_file() and p.name!="transfer_inventory.json"]})
        write_json(statusfile,state)
    complete=all(r["status"] in ("complete","reused_verified_source") for r in state["runs"].values())
    state.update(status="complete" if complete else "finished_with_failures",completed_utc=stamp(),current_run=None,
        elapsed_seconds=time.monotonic()-begin)
    write_json(statusfile,state)


if __name__=="__main__":main()
