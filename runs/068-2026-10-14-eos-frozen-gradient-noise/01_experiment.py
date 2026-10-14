"""Finite-sequence sampling controls on complete real-data objectives."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from src.eos_tasks.task import Task,configure,model_for
from src.cifar_eos.core import (Geometry,append_json,environment,file_hash,memory,stamp,synchronize,tensor_hash,write_json)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--mlp-cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    inputs=json.loads((a.cache/"inputs.json").read_text())
    assert file_hash(a.cache/"inputs.json")==cfg["inputs_sha256"]
    configure()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest={"status":"running","started_utc":stamp(),"config_sha256":file_hash(a.config),
        "inputs_sha256":file_hash(a.cache/"inputs.json"),"environment":environment("cuda:0"),
        "source_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in
             [HERE/"01_experiment.py"]+list((ROOT/"src/eos_tasks").glob("*.py"))+list((ROOT/"src/cifar_eos").glob("*.py"))},
        "optimization_updates":0,"frozen_state":"initialization"}
    write_json(a.output/"manifest.json",manifest)
    clock=time.perf_counter()
    conditions=[]
    for name in cfg["tasks"]:
        datafile=a.mlp_cache/"data.pt" if name=="mlp" else a.cache/(name+"_data.pt")
        data={k:v.cuda() for k,v in torch.load(datafile,weights_only=True).items() if k.startswith(("x_","y_"))}
        for seed in cfg["seeds"]:
            model=model_for(name,a.cache,inputs["wikitext"]["vocabulary_size"]).cuda()
            initial=a.cache/f"{name}_initial_seed{seed}.pt"
            assert file_hash(initial)==inputs["initial"][name][seed]["sha256"]
            model.load_state_dict(torch.load(initial,weights_only=True))
            before=tensor_hash(model.parameters())
            task=Task(name,model,data,64 if name=="transformer" else 1000)
            print(json.dumps({"phase":"full_reference","task":name,"seed":seed,"utc":stamp()}),flush=True)
            loss,reference=task.gradient()
            reference=reference.double()
            signal_norm=float(reference.norm())
            if signal_norm==0:raise RuntimeError("Frozen full gradient has zero norm; control undefined.")
            pool=None
            if name=="resnet":
                # All50 objective units can be stored, so every sampled batch
                # gradient is a real-data average with unchanged BN context.
                pool=torch.stack([task.gradient(torch.tensor([i],device="cuda"))[1] for i in range(task.n)])
                discrepancy=float((pool.double().mean(0)-reference).norm())
                assert discrepancy<=2e-5*max(1,signal_norm)
            for b in cfg["batch_units"][name]+[task.n]:
                tag=f"{name}-seed{seed}-b{b}"
                full=b==task.n
                # Pair conditions using a fresh permutation stream for every
                # draw; reset perbatch to share prefix indices across sizes.
                rng=np.random.default_rng(cfg["sampling_seed"]+seed)
                prefix=a.output/tag
                prefix.mkdir()
                replicate=[]
                started=time.perf_counter()
                maximum_identity_error=0.
                with (prefix/"batch-indices.u32").open("wb") as draws:
                    for rep in range(cfg["replicates"]):
                        stream=Geometry(len(reference),reference.device)
                        previous_noise=None
                        energies=[];cosines=[];qs=[];norms=[]
                        for step in range(cfg["sequence_length"]):
                            if full:g=reference
                            else:
                                indices=rng.permutation(task.n)[:b].astype("<u4")
                                draws.write(indices.tobytes())
                                ids=torch.from_numpy(indices.astype(np.int64)).cuda()
                                if pool is None:_,g=task.gradient(ids)
                                else:g=pool[ids].double().mean(0)
                                g=g.double()
                            noise=g-reference
                            previous=stream.previous
                            denominator=None if previous is None else float(previous.square().sum())
                            q=None if not denominator else 1-float(previous.dot(g))/denominator
                            epsilon=None if not denominator else float(previous.dot(noise-previous_noise))/denominator
                            if q is not None:
                                error=abs(q+epsilon)
                                maximum_identity_error=max(maximum_identity_error,error)
                                assert error<2e-9*max(1,abs(q),abs(epsilon))
                                qs.append(q)
                            stats=stream.consume(g)
                            energy=float(noise.square().sum())
                            energies.append(energy);norms.append(stats["norm"])
                            if stats["cosine"] is not None:cosines.append(stats["cosine"])
                            append_json(prefix/"metrics.jsonl",{"replicate":rep,"draw":step+1,
                                "norm":stats["norm"],"cosine":stats["cosine"],"A":stats["A"],"B":stats["B"],
                                "B_over_A":stats["B_over_A"],"noise_energy":energy,
                                "gradient_only_projected_change":q,"noise_epsilon":epsilon,
                                "noise_corrected_projected_change":None if q is None else q+epsilon})
                            previous_noise=noise
                        replicate.append({"replicate":rep,"B_over_A":stats["B_over_A"],"mean_cosine":float(np.mean(cosines)),
                            "mean_norm":float(np.mean(norms)),"mean_noise_energy":float(np.mean(energies)),
                            "mean_gradient_only_projected_change":float(np.mean(qs))})
                        if rep%4==3:
                            print(json.dumps({"phase":"frozen_probe","condition":tag,"replicate":rep+1,
                                "replicates":cfg["replicates"],"elapsed_seconds":time.perf_counter()-clock,**memory("cuda:0")}),flush=True)
                assert tensor_hash(model.parameters())==before,"Frozen probes changed parameters."
                summary={"task":name,"seed":seed,"batch_units":b,"population_units":task.n,"full_batch":full,
                    "objective_loss":loss,"signal_norm":signal_norm,"initial_tensor_sha256":before,
                    "replicates":replicate,"mean_B_over_A":float(np.mean([r["B_over_A"] for r in replicate])),
                    "mean_cosine":float(np.mean([r["mean_cosine"] for r in replicate])),
                    "mean_noise_energy":float(np.mean([r["mean_noise_energy"] for r in replicate])),
                    "noise_energy_monte_carlo_se":float(np.std([r["mean_noise_energy"] for r in replicate],ddof=1)/np.sqrt(cfg["replicates"])),
                    "maximum_corrected_identity_error":maximum_identity_error,"seconds":time.perf_counter()-started}
                summary["relative_noise_rms"]=float(np.sqrt(summary["mean_noise_energy"]))/signal_norm
                write_json(prefix/"summary.json",summary)
                conditions.append(summary)
                write_json(a.output/"results.json",{"conditions":conditions,"status":"running"})
            del task,model,reference,pool
            torch.cuda.empty_cache()
        del data
    write_json(a.output/"results.json",{"conditions":conditions,"status":"complete"})
    manifest.update(status="complete",completed_utc=stamp(),elapsed_seconds=time.perf_counter()-clock,memory=memory("cuda:0"),
        files={p.relative_to(a.output).as_posix():{"bytes":p.stat().st_size,"sha256":file_hash(p)}
            for p in a.output.rglob("*") if p.is_file() and p.name!="manifest.json"})
    write_json(a.output/"manifest.json",manifest)


if __name__=="__main__":main()
