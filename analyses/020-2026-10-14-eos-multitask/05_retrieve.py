"""Retrieve terminal task-owned artifacts, verifying every retained byte/hash."""
import argparse
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,write_json
HERE=Path(__file__).parent
HOST="tcml-spark@spark-2277.local"
REMOTE="/home/tcml-spark/experiments/constructive-destructive-regimes/eos-multitask-2026-10-14"
PYTHON="/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv/bin/python"


def remote_python(code):
    return subprocess.check_output(["ssh","-o","BatchMode=yes",HOST,shlex.join([PYTHON,"-c",code])],text=True,timeout=600)


def retrieve(run):
    local=ROOT/"runs"/run/"artifacts/attempt-01-spark"
    output=REMOTE+"/runs/"+run+"/artifacts/attempt-01-spark"
    code="import pathlib,json,hashlib,tarfile; "+f"root=pathlib.Path({output!r}); "+\
        "m=json.loads((root/'production/manifest.json').read_text()); assert m['status']=='complete'; "+\
        "files=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='transfer_inventory.json']; "+\
        "(root/'transfer_inventory.json').write_text(json.dumps({'files':files},indent=2)+'\\n'); "+\
        "bundle=root.parent/'retrieval.tar'; t=tarfile.open(bundle,'w'); t.add(root,arcname='attempt-01-spark'); t.close(); "+\
        "print(json.dumps({'bytes':bundle.stat().st_size,'sha256':hashlib.file_digest(bundle.open('rb'),'sha256').hexdigest(),'path':str(bundle)}))"
    archive=json.loads(remote_python(code))
    target=ROOT/"cache"/(run+"-retrieval.tar")
    subprocess.run(["scp","-q",HOST+":"+archive["path"],str(target)],check=True)
    assert target.stat().st_size==archive["bytes"] and file_hash(target)==archive["sha256"]
    local.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target) as bundle:
        for member in bundle.getmembers():
            destination=(local.parent/member.name).resolve()
            assert destination.is_relative_to(local.parent.resolve()),member.name
        bundle.extractall(local.parent,filter="data")
    inventory=json.loads((local/"transfer_inventory.json").read_text())
    for item in inventory["files"]:
        path=local/item["path"]
        assert path.resolve().is_relative_to(local.resolve())
        assert path.stat().st_size==item["bytes"] and file_hash(path)==item["sha256"],item["path"]
    receipt={"status":"passed","run":run,"files":len(inventory["files"]),"retained_bytes":sum(i["bytes"] for i in inventory["files"]),
        "inventory_sha256":file_hash(local/"transfer_inventory.json"),"archive":archive}
    write_json(ROOT/"runs"/run/"retrieval_verification.json",receipt)
    # Preserve the exact executed source packet in the owning attempt.
    if run.startswith("068-"):
        snapshot=local/"source"
        snapshot.mkdir(exist_ok=True)
        with tarfile.open(ROOT/"cache/eos-frozen-noise-sources.tar") as bundle:
            bundle.extractall(snapshot,filter="data")
    # The archives contain no unique evidence after both copies/inventories
    # pass. Remove only this task's verified temporary transfer files so raw
    # checkpoints plus redundant bundles stay within the storage allowance.
    assert target.resolve().is_relative_to((ROOT/"cache").resolve())
    target.unlink()
    cleanup="import pathlib; "+f"p=pathlib.Path({archive['path']!r}).resolve(); root=pathlib.Path({output!r}).parent.resolve(); "+\
        "assert p.is_relative_to(root) and p.name=='retrieval.tar'; p.unlink()"
    remote_python(cleanup)
    receipt["archive"]["retained"]=False
    receipt["archive"]["note"]="Verified temporary archives removed; raw artifacts and source snapshots retained."
    write_json(ROOT/"runs"/run/"retrieval_verification.json",receipt)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",action="store_true")
    a=p.parse_args()
    if a.frozen:
        receipt=retrieve("068-2026-10-14-eos-frozen-gradient-noise")
        write_json(HERE/"frozen_retrieval.json",receipt)
        print(json.dumps(receipt,indent=2))
    else:
        cohort=json.loads((HERE/"cohort.json").read_text())
        receipts=[]
        for run in cohort["runs"]:
            if run["kind"]=="new_training":receipts.append(retrieve(run["run"]))
        write_json(HERE/"retrieval_verification.json",{"status":"passed","runs":receipts})
        print(json.dumps({"status":"passed","runs":len(receipts)}))


if __name__=="__main__":main()
