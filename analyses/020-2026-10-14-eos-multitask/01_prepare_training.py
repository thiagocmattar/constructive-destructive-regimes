"""Make the reviewable training plan; freeze it only after horizon selection."""
import argparse
import json
import sys
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json

REMOTE="/home/tcml-spark/experiments/constructive-destructive-regimes/eos-multitask-2026-10-14"
PYTHON="/home/tcml-spark/experiments/constructive-destructive-regimes/spark-calibration-2026-10-08/venv/bin/python"
DEPS="/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/dependencies"
MLP="/home/tcml-spark/experiments/constructive-destructive-regimes/cifar-eos-calibration-2026-10-13/cache/cifar10-eos"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--horizon",choices=["staged-500","paper-length"],required=True)
    p.add_argument("--plan-only",action="store_true")
    a=p.parse_args()
    if (HERE/"cohort.json").exists():raise FileExistsError("Scientific cohort already frozen.")
    if json.loads((HERE/"mathematical_verification.json").read_text())["status"]!="passed":raise RuntimeError("Verification missing.")
    inputs=json.loads((ROOT/"cache/eos-multitask/inputs.json").read_text())
    calibration=ROOT/"runs/067-2026-10-14-eos-multitask-calibration/artifacts/attempt-01-spark"
    rates={"mlp":.01,"resnet":.2,"transformer":.1}
    batches={"mlp":[32,128,512],"resnet":[1,5,20],"transformer":[8,32,128]}
    updates={t:500 for t in rates} if a.horizon=="staged-500" else {"mlp":20000,"resnet":2000,"transformer":15000}
    records=[]
    number=69
    total_seconds=0
    files=[]
    for method in ("gd","sgd"):
        for task in rates:
            bench=json.loads((calibration/task/"benchmark.json").read_text())
            assert bench["status"]=="complete"
            full_seconds=bench["samples"][-1]["seconds"]
            hessian_seconds=bench["hessian"]["seconds"]
            if task=="mlp":hessian_seconds*=2 # calibration2.5k; production Hessian uses all5k
            for b in ([None] if method=="gd" else batches[task]):
                for seed in (0,1,2):
                    if method=="gd" and task=="mlp" and seed==0:
                        records.append({"run_id":"061","kind":"reused_gd_prefix","task":"mlp","method":"gd","seed":0,
                            "lr":rates[task],"updates":updates[task],"run":"061-2026-10-13-cifar10-eos-gd-lr-1e-2",
                            "output":"artifacts/attempt-01-spark/production","source":"verified immutable Analysis019 Run061"})
                        continue
                    label="full" if b is None else f"b{b}"
                    name=f"{number:03}-2026-10-14-eos-{task}-{method}-{label}-seed{seed}"
                    hessian_every=100 if method=="gd" or a.horizon=="staged-500" else 500
                    expected=updates[task]*full_seconds if b is None else updates[task]*full_seconds*b/bench["objective_units"]
                    expected+=(updates[task]//hessian_every+1)*hessian_seconds
                    if method=="sgd":
                        expected+=2*(updates[task]//100+1)*full_seconds
                        if task=="resnet":expected+=(updates[task]//100+1)*full_seconds
                    # Include complete test evaluation, recovery serialization
                    # and per-update vector geometry, absent in backward timing.
                    expected+=updates[task]*.015+(updates[task]//500+1)*full_seconds*.25+20
                    total_seconds+=expected
                    cfg={"run_id":f"{number:03}","hypothesis":"H019","task":task,"method":method,"seed":seed,
                        "lr":rates[task],"batch_units":b,"updates":updates[task],"horizon":a.horizon,
                        "physical":64 if task=="transformer" else 1000,"noise_every":100,"variance_sample_units":32,
                        "hessian_units":5000 if task=="mlp" else (5 if task=="resnet" else 2500),
                        "hessian_every":hessian_every,"checkpoint_every":100,"test_every":500,
                        "retained_states":[updates[task]//2],"eigensolver":{"k":1,"which":"LA","tolerance":.001,
                            "ncv":12,"maxiter":200,"residual_tolerance":.005,"fallback":{"ncv":36,"maxiter":400}},
                        "input_manifest_sha256":file_hash(ROOT/"cache/eos-multitask/inputs.json"),
                        "notes":["Constant-step plain GD/SGD; no momentum, decay, clipping, schedule, augmentation or dropout.",
                            "Every raw adjacent pair and complete-prefix B/A recorded; noise-corrected adjacent pairs at sparse declared probes.",
                            "ResNet samples fixed1k ghost groups; Transformer samples fixed causal sequences with token-weighted unbiased gradients.",
                            "Largest algebraic Hessian eigenvalue; subset coverage explicit; fresh residual check.",
                            "Initial and retained/recovery/final states; RNG/diagnostics retained; no dense training-gradient histories."]}
                    record={"run_id":f"{number:03}","run":name,"kind":"new_training","task":task,"method":method,
                        "seed":seed,"lr":rates[task],"batch_units":b,"updates":updates[task],
                        "estimated_seconds":expected,"maximum_seconds":max(1800,int(expected*1.6+600)),"config":cfg}
                    records.append(record)
                    if not a.plan_only:
                        folder=ROOT/"runs"/name
                        folder.mkdir(exist_ok=False)
                        write_json(folder/"config.json",cfg)
                        (folder/"README.md").write_text(f"# Run{number:03}: {task} {method.upper()}, seed{seed}\n\n"
                            "User-selected real-data EOS case under the [Analysis020 protocol](../../analyses/020-2026-10-14-eos-multitask/PROTOCOL.md).\n"
                            f"Fixed{updates[task]} updates, LR{rates[task]}, effective batch units{label}.\n"
                            "Config and attempt source are frozen; status lives in the attempt manifest.\n",encoding="utf-8")
                        record["config_sha256"]=file_hash(folder/"config.json")
                        files.append(folder/"config.json")
                    number+=1
    cohort={"analysis_id":"020","hypothesis":"H019","created_utc":stamp(),"horizon":a.horizon,
        "runs":records,"estimated_hours":total_seconds/3600,"qualified_hours":total_seconds*1.4/3600,
        "hard_queue_seconds":72*3600 if a.horizon=="staged-500" else 21*24*3600,
        "remote_root":REMOTE,"remote_python":PYTHON,"remote_dependencies":DEPS,"remote_mlp_cache":MLP,
        "remote_cache":REMOTE+"/cache/eos-multitask","storage_allowance_bytes":40_000_000_000,
        "cost":0,"owned_spark":True,"dense_gradient_histories":False,"next_run":number,
        "limitations":"One prespecified LR per task, three seeds; staged500 is a first-stage result, not a completed paper replication."}
    if a.plan_only:
        write_json(HERE/(a.horizon+"-plan.json"),cohort)
    else:
        write_json(HERE/"cohort.json",cohort)
        files+=list((ROOT/"src/eos_tasks").glob("*.py"))+list((ROOT/"src/cifar_eos").glob("*.py"))
        files+=list(HERE.glob("*.py"))+[HERE/"cohort.json",HERE/"mathematical_verification.json"]
        inventory={"files":[{"path":f.relative_to(ROOT).as_posix(),"bytes":f.stat().st_size,"sha256":file_hash(f)} for f in files]}
        write_json(HERE/"source_inventory.json",inventory)
        packet=ROOT/"cache/eos-training-sources.tar"
        with tarfile.open(packet,"w") as bundle:
            for f in files+[HERE/"source_inventory.json"]:bundle.add(f,arcname=f.relative_to(ROOT).as_posix())
        write_json(HERE/"packet.json",{"path":packet.relative_to(ROOT).as_posix(),"bytes":packet.stat().st_size,"sha256":file_hash(packet)})
    print(json.dumps({"horizon":a.horizon,"new_runs":number-69,"reused":1,"estimated_hours":cohort["estimated_hours"],
        "qualified_hours":cohort["qualified_hours"],"hard_queue_hours":cohort["hard_queue_seconds"]/3600,"next_run":number},indent=2))


if __name__=="__main__":main()
