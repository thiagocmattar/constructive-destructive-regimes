"""Calibration jobs survive SSH; each task has its own log and timeout."""
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json


def main():
    for f in json.loads((HERE/"source_inventory.json").read_text())["files"]:
        path=ROOT/f["path"]
        assert path.stat().st_size==f["bytes"] and file_hash(path)==f["sha256"],f["path"]
    state={"status":"running","started_utc":stamp(),"tasks":{}}
    write_json(HERE/"queue.json",state)
    config=json.loads((HERE/"config.json").read_text())
    for name in config["task_order"]:
        state["current"]=name
        state["tasks"][name]={"status":"running","started_utc":stamp()}
        write_json(HERE/"queue.json",state)
        log=HERE/"artifacts/attempt-01-spark"/(name+".log")
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("x") as output:
            args=["timeout",str(config["maximum_seconds_per_task"]),sys.executable,str(HERE/"02_benchmark.py"),
                "--task",name,"--cache",str(ROOT/"cache/eos-multitask"),"--mlp-cache",
                "/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/cache/cifar10-eos",
                "--output",str(log.parent/name),"--physical",str(config["physical_transformer_sequences"])]
            result=subprocess.run(args,stdout=output,stderr=subprocess.STDOUT)
        state["tasks"][name].update(status="complete" if result.returncode==0 else "failed",exit_code=result.returncode,ended_utc=stamp())
        write_json(HERE/"queue.json",state)
    state.update(status="complete" if all(x["status"]=="complete" for x in state["tasks"].values()) else "finished_with_failures",completed_utc=stamp(),current=None)
    write_json(HERE/"queue.json",state)


if __name__=="__main__":main()
