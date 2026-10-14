"""Freeze/transfer a calibration packet; launch a bounded persistent Spark queue."""
import json
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
    receipt=HERE/"launch.json"
    if receipt.exists():raise FileExistsError("Calibration already launched; preserve inputs.")
    config={"task_order":["mlp","resnet","transformer"],"physical_transformer_sequences":64,
        "full_gradient_timing_repetitions":2,"hessian_objective_units":{"mlp":2500,"resnet":5,"transformer":2500},
        "hessian":{"k":1,"which":"LA","tolerance":.001,"residual_tolerance":.005},
        "maximum_seconds_per_task":2400,"maximum_queue_seconds":7200,"rental_cost":0,
        "owned_spark":True,"storage_allowance_bytes":3_000_000_000}
    write_json(HERE/"config.json",config)
    files=list((ROOT/"src/cifar_eos").glob("*.py"))+list(HERE.glob("*.py"))+[HERE/"config.json",HERE/"inputs.json",HERE/"mathematical_verification.json"]
    cache=ROOT/"cache/eos-multitask"
    files+=list(cache.glob("*.pt"))+[cache/"inputs.json"]+list((cache/"reference").glob("*"))
    inventory={"files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":file_hash(p)} for p in files if p.is_file()]}
    write_json(HERE/"source_inventory.json",inventory)
    packet=ROOT/"cache/eos-multitask-calibration.tar"
    with tarfile.open(packet,"w") as bundle:
        for p in files+[HERE/"source_inventory.json"]:
            if p.is_file():bundle.add(p,arcname=p.relative_to(ROOT).as_posix())
    digest=file_hash(packet)
    subprocess.run(["ssh","-o","BatchMode=yes",HOST,"mkdir","-p",REMOTE],check=True)
    subprocess.run(["scp","-q",str(packet),HOST+":"+REMOTE+"/calibration.tar"],check=True)
    remote_digest=subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,"sha256sum",REMOTE+"/calibration.tar"],text=True).split()[0]
    assert remote_digest==digest
    subprocess.run(["ssh","-o","BatchMode=yes",HOST,"tar","-xf",REMOTE+"/calibration.tar","-C",REMOTE],check=True)
    code=f"cd {REMOTE} && PYTHONPATH={REMOTE}:{DEPS} nohup timeout 7200 {PYTHON} runs/{HERE.name}/04_queue.py > calibration-queue.log 2>&1 < /dev/null &"
    subprocess.run(["ssh","-o","BatchMode=yes",HOST,code],check=True,timeout=30)
    write_json(receipt,{"status":"launched","remote_root":REMOTE,"python":PYTHON,"dependencies":DEPS,"mlp_cache":MLP,
        "packet_bytes":packet.stat().st_size,"packet_sha256":digest,"remote_packet_sha256":remote_digest,"config":config})
    print(json.dumps(json.loads(receipt.read_text()),indent=2))


if __name__=="__main__":main()
