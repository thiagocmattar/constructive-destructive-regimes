"""Production-shaped three-update checks; no smoke result is paper evidence."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import write_json,file_hash


def main():
    here=Path(__file__).parent
    folder=here/"scratch/production-smoke"
    folder.mkdir(parents=True,exist_ok=False)
    results=[]
    for task,b,units,lr in [("mlp",32,50,.01),("resnet",1,1,.2),("transformer",8,16,.1)]:
        cfg={"run_id":"mechanics-smoke","task":task,"method":"sgd","seed":0,"lr":lr,"batch_units":b,
            "updates":3,"physical":64 if task=="transformer" else 1000,"noise_every":2,"variance_sample_units":4,
            "hessian_units":units,"hessian_every":2,"checkpoint_every":2,"test_every":3,"retained_states":[2],
            "eigensolver":{"k":1,"tolerance":.001,"ncv":20,"maxiter":200,"residual_tolerance":.005},
            "input_manifest_sha256":file_hash(ROOT/"cache/eos-multitask/inputs.json")}
        config=folder/(task+".json")
        write_json(config,cfg)
        command=[sys.executable,"-m","src.eos_tasks.train","--config",str(config),"--cache",str(ROOT/"cache/eos-multitask"),
            "--mlp-cache",str(ROOT/"cache/cifar10-eos"),"--output",str(folder/task)]
        with (folder/(task+".log")).open("x") as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        output=folder/task
        manifest=json.loads((output/"manifest.json").read_text())
        rows=[json.loads(s) for s in (output/"metrics.jsonl").read_text().splitlines()]
        assert manifest["status"]=="complete" and len(rows)==3
        assert rows[0]["streams"]["raw"]["cosine"] is None
        assert rows[1]["noise_epsilon"] is not None
        for row in rows:
            assert 0<=row["streams"]["raw"]["B_over_A"]<=1+1e-10
            assert row["raw_update_cosine"]<-.99999
            assert row["cumulative_update_identity_residual_norm"]<1e-4
        spectra=[json.loads(s) for s in (output/"curvature.jsonl").read_text().splitlines()]
        assert [s["state_step"] for s in spectra]==[0,2,3]
        assert max(r for s in spectra for r in s["relative_residuals"])<=.005
        results.append({"task":task,"status":"passed","updates":3,"noise_corrected_adjacent_pairs":1,
            "spectra":3,"maximum_update_identity_residual":max(r["cumulative_update_identity_residual_norm"] for r in rows),
            "peak_gpu_bytes":json.loads((output/"summary.json").read_text())["memory"]["peak_allocated_bytes"],
            "smoke_manifest_sha256":file_hash(output/"manifest.json")})
        print(task+" production-shaped smoke passed.",flush=True)
    write_json(here/"smoke_verification.json",{"status":"passed","scientific_evidence":False,"results":results})


if __name__=="__main__":main()
