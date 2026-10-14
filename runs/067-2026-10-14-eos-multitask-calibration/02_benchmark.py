"""Measure full passes and subset-Hessian cost before fixing production budgets."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from eos import Task, configure, model_for
from src.cifar_eos.core import environment,file_hash,memory,stamp,synchronize,tensor_hash,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--mlp-cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task",choices=["mlp","resnet","transformer"],required=True)
    p.add_argument("--physical",type=int,default=64)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    configure()
    manifest=json.loads((a.cache/"inputs.json").read_text())
    datafile=a.mlp_cache/"data.pt" if a.task=="mlp" else a.cache/(a.task+"_data.pt")
    print(json.dumps({"phase":"load","task":a.task,"utc":stamp()}),flush=True)
    data={k:v.cuda() for k,v in torch.load(datafile,weights_only=True).items() if k.startswith(("x_","y_"))}
    model=model_for(a.task,a.cache,manifest["wikitext"]["vocabulary_size"]).cuda()
    model.load_state_dict(torch.load(a.cache/(a.task+"_initial_seed0.pt"),weights_only=True))
    identity=manifest["initial"][a.task][0]
    assert tensor_hash(model.parameters())==identity["initial_tensor_sha256"]
    task=Task(a.task,model,data,a.physical if a.task=="transformer" else 1000)
    torch.cuda.reset_peak_memory_stats()
    record={"task":a.task,"status":"running","environment":environment("cuda:0"),
        "parameter_count":identity["parameters"],"objective_units":task.n,"loss_units":task.units,
        "inputs_sha256":file_hash(a.cache/"inputs.json"),"physical":task.physical,"samples":[]}
    write_json(a.output/"benchmark.json",record)
    for index in range(2):
        print(json.dumps({"phase":"full_gradient","sample":index,"utc":stamp()}),flush=True)
        synchronize("cuda:0");begin=time.perf_counter()
        loss,g=task.gradient()
        synchronize("cuda:0")
        sample={"seconds":time.perf_counter()-begin,"loss":loss,"norm":float(g.double().norm())}
        record["samples"].append(sample)
        write_json(a.output/"benchmark.json",record)
        print(json.dumps(sample),flush=True)
    ids=torch.arange(min(task.n,5 if a.task=="resnet" else 2500),device="cuda")
    print(json.dumps({"phase":"hessian","objective_units":len(ids),"utc":stamp()}),flush=True)
    record["hessian"]=task.sharpness(ids,{"k":1,"tolerance":1e-3,"ncv":12,"maxiter":200,"residual_tolerance":.005})
    print(json.dumps(record["hessian"]),flush=True)
    batch=1 if a.task=="resnet" else (32 if a.task=="transformer" else 32)
    timings=[]
    for i in range(3):
        idx=torch.randperm(task.n,device="cuda")[:batch]
        synchronize("cuda:0");begin=time.perf_counter();task.gradient(idx);synchronize("cuda:0")
        timings.append(time.perf_counter()-begin)
    record.update(status="complete",batch_unit_count=batch,batch_seconds=timings,memory=memory("cuda:0"),completed_utc=stamp())
    write_json(a.output/"benchmark.json",record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=="__main__":main()
