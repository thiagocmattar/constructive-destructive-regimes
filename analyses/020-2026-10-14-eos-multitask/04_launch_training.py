"""Launch only an explicitly frozen horizon; reused authorization is in cohort."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json
HERE=Path(__file__).parent
HOST="tcml-spark@spark-2277.local"


def main():
    if (HERE/"training_launch.json").exists():raise FileExistsError("Training queue already launched.")
    cohort=json.loads((HERE/"cohort.json").read_text())
    assert json.loads((HERE/"smoke_verification.json").read_text())["status"]=="passed"
    remote=cohort["remote_root"]
    python=cohort["remote_python"]
    frozen=remote+"/runs/068-2026-10-14-eos-frozen-gradient-noise/artifacts/attempt-01-spark/production/manifest.json"
    check="import json; print(json.load(open("+repr(frozen)+"))['status'])"
    terminal=subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,shlex.join([python,"-c",check])],text=True).strip()
    assert terminal=="complete","Finish frozen control before sharing the Spark GPU."
    packet=json.loads((HERE/"packet.json").read_text())
    source=ROOT/packet["path"]
    assert file_hash(source)==packet["sha256"]
    subprocess.run(["scp","-q",str(source),HOST+":"+remote+"/training-sources.tar"],check=True)
    remotehash=subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,"sha256sum",remote+"/training-sources.tar"],text=True).split()[0]
    assert remotehash==packet["sha256"]
    subprocess.run(["ssh","-o","BatchMode=yes",HOST,"tar","-xf",remote+"/training-sources.tar","-C",remote],check=True)
    args=[cohort["remote_python"],remote+"/analyses/"+HERE.name+"/03_queue.py"]
    code="import os,pathlib,subprocess,json; "+f"root=pathlib.Path({remote!r}); env=dict(os.environ,PYTHONPATH={remote+':'+cohort['remote_dependencies']!r}); "+\
        "log=(root/'training-queue.log').open('x'); "+f"child=subprocess.Popen({args!r},cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); "+\
        "(root/'training-process.json').write_text(json.dumps({'pid':child.pid,'command':child.args})); print(child.pid)"
    pid=int(subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,shlex.join([python,"-c",code])],text=True,timeout=30).strip())
    write_json(HERE/"training_launch.json",{"status":"launched","utc":stamp(),"pid":pid,"remote_root":remote,
        "horizon":cohort["horizon"],"expected_hours":cohort["estimated_hours"],"qualified_hours":cohort["qualified_hours"],
        "hard_queue_seconds":cohort["hard_queue_seconds"],"packet_sha256":packet["sha256"],"remote_packet_sha256":remotehash,"rental_cost":0})
    print(json.dumps(json.loads((HERE/"training_launch.json").read_text()),indent=2))


if __name__=="__main__":main()
