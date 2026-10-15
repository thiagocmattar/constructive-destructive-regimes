"""Detached local retrieval/plotting controller; never auto-accept visual QA."""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import stamp,write_json
HERE=Path(__file__).parent
HOST="tcml-spark@spark-2277.local"
POLL_SECONDS=30*60
WINDOW_FLAGS=getattr(subprocess,"CREATE_NO_WINDOW",0)


def main():
    cohort=json.loads((HERE/"cohort.json").read_text())
    statusfile=HERE/"local-controller.json"
    started=time.monotonic()
    while time.monotonic()-started<cohort["hard_queue_seconds"]+3600:
        try:
            remote=cohort["remote_root"]+"/analyses/"+HERE.name+"/queue_state.json"
            code="import json; print(json.dumps(json.load(open("+repr(remote)+"))))"
            args=["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",HOST,
                shlex.join([cohort["remote_python"],"-c",code])]
            state=json.loads(subprocess.check_output(args,text=True,timeout=30,creationflags=WINDOW_FLAGS))
            event={"utc":stamp(),"status":state["status"],"current_run":state.get("current_run"),
                "completed_runs":sum(v["status"]=="complete" for v in state["runs"].values()),
                "total_new_runs":sum(r["kind"]=="new_training" for r in cohort["runs"]),
                "poll_seconds":POLL_SECONDS}
            write_json(statusfile,event)
            with (HERE/"monitoring.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(event)+"\n")
            if state["status"]=="complete":
                subprocess.run([sys.executable,str(HERE/"05_retrieve.py")],cwd=ROOT,check=True,creationflags=WINDOW_FLAGS)
                subprocess.run([sys.executable,str(HERE/"07_plot_training.py")],cwd=ROOT,check=True,creationflags=WINDOW_FLAGS)
                write_json(statusfile,{"status":"retrieved_and_plotted_awaiting_visual_review","utc":stamp(),
                    "note":"PDFs and source/array receipts generated; no visual acceptance or manuscript training claims made automatically."})
                return
            if state["status"]=="finished_with_failures":
                write_json(statusfile,{"status":"requires_failure_review","utc":stamp(),"remote_status":state})
                return
        except Exception as error:
            write_json(statusfile,{"status":"retrying_read_or_requires_review","utc":stamp(),"error":repr(error)})
        time.sleep(POLL_SECONDS)
    write_json(statusfile,{"status":"controller_deadline","utc":stamp()})


if __name__=="__main__":main()
