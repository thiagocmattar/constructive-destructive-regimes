"""Constant-step plain GD/SGD with full-prefix geometry and measured noise probes."""
import argparse
import json
import time
from pathlib import Path

import torch

from .noise import variance_probe
from .task import configure, model_for
from .production import ProductionTask as Task
from src.cifar_eos.core import (Geometry,append_json,cosine,environment,file_hash,flat,
    memory,stamp,synchronize,tensor_hash,write_json)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--mlp-cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    inputs=json.loads((args.cache/"inputs.json").read_text())
    if "input_manifest_sha256" in cfg:
        assert file_hash(args.cache/"inputs.json")==cfg["input_manifest_sha256"]
    for name,identity in inputs["reference_sources"].items():
        assert file_hash(args.cache/"reference"/name)==identity["sha256"]
    configure()
    device=torch.device("cuda:0")
    datafile=args.mlp_cache/"data.pt" if cfg["task"]=="mlp" else args.cache/(cfg["task"]+"_data.pt")
    if cfg["task"]=="mlp":
        old_inputs=json.loads((args.mlp_cache/"inputs.json").read_text())
        assert file_hash(datafile)==old_inputs["prepared_files"]["data.pt"]["sha256"]
    else:
        assert file_hash(datafile)==inputs["prepared_files"][datafile.name]["sha256"]
    data={k:v.to(device) for k,v in torch.load(datafile,weights_only=True).items() if k.startswith(("x_","y_"))}
    model=model_for(cfg["task"],args.cache,inputs["wikitext"]["vocabulary_size"]).to(device)
    initialfile=args.cache/f'{cfg["task"]}_initial_seed{cfg["seed"]}.pt'
    model.load_state_dict(torch.load(initialfile,weights_only=True))
    expected=inputs["initial"][cfg["task"]][cfg["seed"]]
    assert file_hash(initialfile)==expected["sha256"]
    assert tensor_hash(model.parameters())==expected["initial_tensor_sha256"]
    task=Task(cfg["task"],model,data,cfg["physical"])
    b=task.n if cfg["batch_units"] is None else cfg["batch_units"]
    assert 1<=b<=task.n
    full=b==task.n
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg["lr"],momentum=0,weight_decay=0,foreach=False)
    streams={name:Geometry(expected["parameters"],device) for name in ("raw","displacement")}
    training_rng=torch.Generator(device=device).manual_seed(10000+cfg["seed"])
    probe_rng=torch.Generator(device=device).manual_seed(20000+cfg["seed"])
    start=0
    previous_signal=previous_noise=None
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        assert saved["config_sha256"]==file_hash(args.config)
        assert saved["inputs_sha256"]==file_hash(args.cache/"inputs.json")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        for name,s in streams.items():s.load_state_dict(saved["streams"][name])
        training_rng.set_state(saved["training_rng"].cpu());probe_rng.set_state(saved["probe_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        start=saved["step"]
        previous_signal=saved["previous_signal"];previous_noise=saved["previous_noise"]
    args.output.mkdir(parents=True,exist_ok=False)
    source={p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    manifest={"status":"running","started_utc":stamp(),"config_sha256":file_hash(args.config),
        "inputs_sha256":file_hash(args.cache/"inputs.json"),"data_sha256":file_hash(datafile),
        "initial_sha256":file_hash(initialfile),"initial_tensor_sha256":expected["initial_tensor_sha256"],
        "source":source,"environment":environment(device),"objective_units":task.n,"loss_units":task.units,
        "batch_units":b,"full_batch":full,"requested_updates":cfg["updates"],"start_step":start}
    write_json(args.output/"manifest.json",manifest)
    write_json(args.output/"resolved_config.json",cfg)
    torch.cuda.reset_peak_memory_stats()
    clock=time.perf_counter()
    last_progress=clock
    def save(step,label):
        if step==0:
            write_json(args.output/"initial_state.json",{"step":0,"initial_sha256":manifest["initial_sha256"],
                "initial_file":str(initialfile),"config_sha256":manifest["config_sha256"],
                "training_seed":10000+cfg["seed"],"probe_seed":20000+cfg["seed"],
                "diagnostic_initialization":"all prefix vectors and energies zero; adjacent vectors undefined"})
            return
        path=args.output/(label+".tmp")
        torch.save({"step":step,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "streams":{k:s.state_dict() for k,s in streams.items()},"config_sha256":manifest["config_sha256"],
            "inputs_sha256":manifest["inputs_sha256"],
            "training_rng":training_rng.get_state(),"probe_rng":probe_rng.get_state(),
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "previous_signal":previous_signal,"previous_noise":previous_noise},path)
        path.replace(args.output/(label+".pt"))
    def spectrum(step):
        count=cfg["hessian_units"]
        ids=None if count==task.n else torch.arange(count,device=device)
        print(json.dumps({"phase":"hessian_started","state_step":step,"objective_units":count,"utc":stamp()}),flush=True)
        event=task.sharpness(ids,cfg["eigensolver"])
        event.update(state_step=step,eta_lambda_max=cfg["lr"]*event["eigenvalues"][0],utc=stamp())
        append_json(args.output/"curvature.jsonl",event)
        print(json.dumps({"phase":"hessian",**event}),flush=True)
    save(start,"initial" if start==0 else "resumed")
    spectrum(start)
    for index in range(start,cfg["updates"]):
        begin=time.perf_counter()
        signal=noise=None
        ref_due=not full and index%cfg["noise_every"] in (0,1)
        if ref_due:
            objective,signal=task.gradient()
            if index%cfg["noise_every"]==0:
                # Probe at the reference's exact parameter state. Probe
                # backwards may replace p.grad; the actual batch is evaluated
                # afterwards so SGD always uses its own recorded raw gradient.
                probe=variance_probe(task,probe_rng,signal,b,cfg["variance_sample_units"])
                probe.update(state_step=index,utc=stamp())
                append_json(args.output/"noise.jsonl",probe)
        ids=None if full else torch.randperm(task.n,device=device,generator=training_rng)[:b]
        train_loss,g=task.gradient(ids)
        raw=g.double()
        previous=streams["raw"].previous
        denom=None if previous is None else float(previous.square().sum())
        q=None if denom is None or denom==0 else 1-float(previous.dot(raw))/denom
        if full:objective=train_loss
        if signal is not None:noise=raw-signal.double()
        epsilon=corrected=None
        if ref_due and previous_signal is not None and previous_noise is not None and denom:
            epsilon=float(previous.dot(noise-previous_noise))/denom
            corrected=q+epsilon
            projected=float(previous.dot(previous_signal.double()-signal.double()))/denom
            assert abs(corrected-projected)<=2e-9*max(1,abs(corrected))
        before=flat(model.parameters()).detach().double()
        if not bool(torch.isfinite(raw).all()):raise RuntimeError("Nonfinite gradient.")
        optimizer.step()
        after=flat(model.parameters()).detach().double()
        displacement=after-before
        stats={"raw":streams["raw"].consume(raw),"displacement":streams["displacement"].consume(displacement)}
        rounding=float((displacement+cfg["lr"]*raw).norm())
        row={"update":index+1,"gradient_state_step":index,"batch_objective_estimate":train_loss,
            "full_objective_loss":objective if full or ref_due else None,"full_objective_state_step":index if full or ref_due else None,
            "streams":stats,"eta_kappa_gradient_proxy":q,"noise_epsilon":epsilon,"eta_kappa_noise_corrected":corrected,
            "realized_noise_energy":None if noise is None else float(noise.square().sum()),
            "signal_norm":None if signal is None else float(signal.double().norm()),
            "signal_adjacent_cosine_at_pair":cosine(previous_signal,signal) if previous_signal is not None and signal is not None else None,
            "noise_adjacent_cosine_at_pair":cosine(previous_noise,noise) if previous_noise is not None and noise is not None else None,
            "parameter_norm":float(after.norm()),"gradient_dot_update":float(raw.dot(displacement)),
            "raw_update_cosine":cosine(raw,displacement),"gd_update_rounding_norm":rounding,
            "parameter_displacement_norm":stats["displacement"]["B"],
            "eta_gradient_sum_norm":cfg["lr"]*stats["raw"]["B"],
            "cumulative_update_identity_residual_norm":float((streams["displacement"].total+cfg["lr"]*streams["raw"].total).norm()),
            "sampled_objective_units":(index+1)*b,"seconds":time.perf_counter()-begin}
        append_json(args.output/"metrics.jsonl",row)
        previous_signal=signal.clone() if signal is not None else None
        previous_noise=noise.clone() if noise is not None else None
        step=index+1
        if step%cfg["hessian_every"]==0:spectrum(step)
        if step%cfg["checkpoint_every"]==0:save(step,"recovery")
        if step in cfg["retained_states"]:save(step,f"state-{step:06d}")
        if step%cfg["test_every"]==0:
            append_json(args.output/"evaluation.jsonl",{"state_step":step,"test":task.evaluate("test"),"utc":stamp()})
        if time.perf_counter()-last_progress>=45 or step==cfg["updates"]:
            elapsed=time.perf_counter()-clock
            event={"utc":stamp(),"completed_updates":step,"requested_updates":cfg["updates"],
                "train_loss":train_loss,"elapsed_seconds":elapsed,"updates_per_second":(step-start)/elapsed,
                "remaining_seconds_at_observed_rate":(cfg["updates"]-step)*elapsed/max(1,step-start),**memory(device)}
            append_json(args.output/"events.jsonl",event);print(json.dumps(event),flush=True);last_progress=time.perf_counter()
    if cfg["updates"]%cfg["hessian_every"]:spectrum(cfg["updates"])
    save(cfg["updates"],"final")
    summary={"completed_updates":cfg["updates"],"elapsed_seconds":time.perf_counter()-clock,
        "final_train":task.evaluate(),"final_test":task.evaluate("test"),"memory":memory(device),
        "final_tensor_sha256":tensor_hash(model.parameters()),"final_raw_geometry":stats["raw"]}
    write_json(args.output/"summary.json",summary)
    manifest.update(status="complete",completed_utc=stamp(),completed_updates=cfg["updates"],
        files={p.name:{"bytes":p.stat().st_size,"sha256":file_hash(p)} for p in args.output.iterdir() if p.is_file() and p.name!="manifest.json"})
    write_json(args.output/"manifest.json",manifest)


if __name__=="__main__":main()
