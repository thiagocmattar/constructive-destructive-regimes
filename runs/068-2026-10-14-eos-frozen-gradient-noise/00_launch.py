"""Verified small source packet; detached launch with closed SSH descriptors."""
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,write_json

HOST="tcml-spark@spark-2277.local"
REMOTE="/home/tcml-spark/experiments/constructive-destructive-regimes/eos-multitask-2026-10-14"
PYTHON="/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv/bin/python"
DEPS="/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/dependencies"
MLP="/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/cache/cifar10-eos"


def main():
    if (HERE/"launch.json").exists():raise FileExistsError("Run already launched.")
    cfg={"run_id":"068","hypothesis":"H019","type":"frozen initial-state real-data gradient sampling",
        "tasks":["mlp","resnet","transformer"],"seeds":[0,1,2],"sampling_seed":20261014,
        "batch_units":{"mlp":[32,128,512],"resnet":[1,5,20],"transformer":[8,32,128]},
        "replicates":32,"sequence_length":128,"full_batch_control":True,"optimization_updates":0,
        "inputs_sha256":file_hash(ROOT/"cache/eos-multitask/inputs.json"),"max_seconds":5400,
        "storage_allowance_bytes":500_000_000,"rental_cost":0,"owned_spark":True,
        "precision":"FP32 model/gradient; FP64 reductions and accumulation; TF32 disabled",
        "resnet_sampling":"whole fixed 1000-image ghost groups; full objective preserved",
        "noise_measurement":"actual squared batch-gradient error against complete full-objective gradient",
        "not_training_evidence":True}
    write_json(HERE/"config.json",cfg)
    files=list(HERE.glob("*.py"))+[HERE/"config.json"]+list((ROOT/"src/eos_tasks").glob("*.py"))
    inventory={"files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":file_hash(p)} for p in files]}
    write_json(HERE/"source_inventory.json",inventory)
    packet=ROOT/"cache/eos-frozen-noise-sources.tar"
    with tarfile.open(packet,"w") as bundle:
        for p in files+[HERE/"source_inventory.json"]:bundle.add(p,arcname=p.relative_to(ROOT).as_posix())
    subprocess.run(["scp","-q",str(packet),HOST+":"+REMOTE+"/frozen-sources.tar"],check=True)
    remote_hash=subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,"sha256sum",REMOTE+"/frozen-sources.tar"],text=True).split()[0]
    assert remote_hash==file_hash(packet)
    subprocess.run(["ssh","-o","BatchMode=yes",HOST,"tar","-xf",REMOTE+"/frozen-sources.tar","-C",REMOTE],check=True)
    output=REMOTE+"/runs/"+HERE.name+"/artifacts/attempt-01-spark"
    argv=["timeout",str(cfg["max_seconds"]),PYTHON,REMOTE+"/runs/"+HERE.name+"/01_experiment.py",
        "--config",REMOTE+"/runs/"+HERE.name+"/config.json","--cache",REMOTE+"/cache/eos-multitask",
        "--mlp-cache",MLP,"--output",output+"/production"]
    code="import os,pathlib,subprocess,json; "+f"root=pathlib.Path({output!r}); root.mkdir(parents=True,exist_ok=False); "+\
        f"env=dict(os.environ,PYTHONPATH={REMOTE+':'+DEPS!r}); log=(root/'production.log').open('x'); "+\
        f"child=subprocess.Popen({argv!r},cwd={REMOTE!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); "+\
        "(root/'process.json').write_text(json.dumps({'pid':child.pid,'command':child.args})); print(child.pid)"
    pid=int(subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,shlex.join([PYTHON,"-c",code])],text=True,timeout=30).strip())
    write_json(HERE/"launch.json",{"status":"launched","pid":pid,"remote_root":REMOTE,"output":output,
        "packet_sha256":file_hash(packet),"remote_packet_sha256":remote_hash,"packet_bytes":packet.stat().st_size,
        "max_seconds":cfg["max_seconds"],"expected_minutes":25,"qualified_minutes":45,"cost":0})
    print(json.dumps(json.loads((HERE/"launch.json").read_text()),indent=2))


if __name__=="__main__":main()
