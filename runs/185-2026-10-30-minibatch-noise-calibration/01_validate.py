"""Mechanics checks and a bounded production-shaped calibration, not paper evidence."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('noise_experiment',ROOT/'runs/186-2026-10-30-mlp-minibatch-noise/01_experiment.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)


def math_checks(device):
    ex.configure();torch.manual_seed(192)
    model=ex.make_model(input_size=6,widths=(4,3)).to(device).double()
    data={'x_train':torch.randn(7,6,device=device,dtype=torch.float64),
          'y_train':torch.tensor([0,2,1,3,8,9,5],device=device)}
    task=ex.Task('mlp',model,data,3)
    _,mu=ex.gradient(task)
    per=torch.stack([ex.gradient(task,torch.tensor([i],device=device))[1] for i in range(7)])
    energy=ex.mlp_unit_energy(task)
    checks={'population_mean':float((per.mean(0)-mu).abs().max()),
            'per_example_energy':abs(energy-float(per.square().sum()))}
    cov=ex.covariance(task,mu)['trace_population_covariance']
    for b in (1,3,7):
        samples=torch.stack([per[list(ids)].mean(0) for ids in itertools.combinations(range(7),b)])
        noise=samples-mu
        V=float(noise.square().sum(1).mean())
        expected=(7-b)/(b*6)*cov
        projection=float((samples@noise.T).mean()-torch.sum(samples*noise,dim=1).mean())
        units=samples/samples.norm(dim=1)[:,None]
        checks[f'variance_b{b}']=abs(V-expected)
        checks[f'projection_b{b}']=abs(projection+V)
        checks[f'cosine_b{b}']=abs(float((units@units.T).mean()-units.mean(0).square().sum()))
    g=per[0].clone();product=task.hvp(g)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    eps=1e-5
    ex.step(model,g,-eps);_,plus=ex.gradient(task)
    model.load_state_dict(initial);ex.step(model,g,eps);_,minus=ex.gradient(task)
    model.load_state_dict(initial)
    checks['hvp_finite_difference_relative']=float((product-(plus-minus)/(2*eps)).norm()/product.norm())
    prev_mu=mu;prev_g=g;prev_noise=g-mu
    ex.step(model,g,.001);_,new_mu=ex.gradient(task);new_g=ex.gradient(task,torch.tensor([2,3],device=device))[1]
    pair=ex.noise_pair(prev_g,new_g,prev_noise,new_g-new_mu,prev_mu,new_mu)
    checks['moving_noise_identity']=abs(pair['correction_identity_residual'])
    stream=ex.Geometry(g.numel(),device)
    for v in per:stats=stream.consume(v)
    direct_pairs=sum(float(per[i].dot(per[j]))*2 for i in range(7) for j in range(i))
    direct_lag=sum(float(per[i].dot(per[i-1]))*2 for i in range(1,7))
    checks['all_pair_accounting']=abs(stats['all_pair']-direct_pairs)
    checks['lag_one_accounting']=abs(stats['lag_one']-direct_lag)
    resumed=ex.Geometry(g.numel(),device);resumed.load_state_dict(stream.state_dict())
    a,b=stream.consume(g),resumed.consume(g)
    checks['geometry_recovery']=max(abs(a[k]-b[k]) for k in a if isinstance(a[k],(int,float)))
    assert max(checks.values())<1e-7,checks
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--math-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checks=math_checks(args.device)
    report={'status':'passed','math_checks':checks,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None}
    if not args.math_only:
        cfg=json.loads((ROOT/'runs/186-2026-10-30-mlp-minibatch-noise/config.json').read_text())
        small=dict(cfg,reference_updates=8,checkpoints=[0,8],frozen_repetitions=2,frozen_draws=12,
                   moving_repetitions=1,continuation_updates=12,hvp_predecessor_gradients=[1,7],recovery_every=6)
        ex.write_json(args.output/'calibration_config.json',small)
        runner=ex.Experiment(small,args.output/'smoke',args.device)
        start=time.perf_counter();runner.reference(0);reference_seconds=time.perf_counter()-start
        runner.cfg['checkpoints']=[0]
        start=time.perf_counter();runner.frozen(0,0);frozen_seconds=time.perf_counter()-start
        moving_times={}
        for batch in cfg['batches']:
            start=time.perf_counter();runner.moving(0,0,batch,0)
            moving_times[str(batch)]=time.perf_counter()-start
        # A second call must skip completed trajectories and preserve output bytes.
        dest=runner.out/'moving-seed0-state0-b32-rep00'
        before=ex.file_hash(dest/'metrics.jsonl');runner.moving(0,0,32,0)
        assert before==ex.file_hash(dest/'metrics.jsonl')
        # Resume an identical fork from six accepted updates and compare every scalar.
        resume_cfg=dict(small,continuation_updates=6)
        resumed=ex.Experiment(resume_cfg,args.output/'resume-check',args.device)
        ref=resumed.out/'reference-seed0';ref.mkdir(exist_ok=True)
        shutil.copyfile(runner.out/'reference-seed0/state-0000.pt',ref/'state-0000.pt')
        resumed.moving(0,0,32,0)
        resume_dir=resumed.out/'moving-seed0-state0-b32-rep00'
        (resume_dir/'summary.json').unlink()
        resumed.cfg['continuation_updates']=12
        resumed.moving(0,0,32,0)
        assert (resume_dir/'metrics.jsonl').read_bytes()==(dest/'metrics.jsonl').read_bytes()
        report['six_step_recovery_exact_scalar_replay']=True
        model=ex.load_model(ROOT/cfg['initial_paths']['0'],args.device);task=runner.task(model)
        samples={}
        for b in cfg['batches']:
            timing=[]
            for j in range(28):
                ids=None if b==5000 else torch.arange(b,device=args.device)
                torch.cuda.synchronize();t=time.perf_counter();ex.gradient(task,ids);torch.cuda.synchronize()
                if j>=4:timing.append(time.perf_counter()-t)
            samples[str(b)]=dict(median=float(np.median(timing)),p90=float(np.percentile(timing,90)),n=len(timing))
        # Conservative calibration includes per-fork state writing and overweights HVP frequency.
        projected_moving=sum(moving_times[str(b)]/12*200*9*(1 if b==5000 else 8) for b in cfg['batches'])
        projected_frozen=frozen_seconds/(5*2*12)*(9*5*32*128)
        projected_reference=reference_seconds/8*6000
        report.update(reference_seconds=reference_seconds,frozen_seconds=frozen_seconds,
            moving_seconds=moving_times,gradient_samples=samples,
            projected_seconds=dict(reference=projected_reference,frozen=projected_frozen,moving=projected_moving,
                total=projected_reference+projected_frozen+projected_moving),
            peak_allocated_gb=torch.cuda.max_memory_allocated(args.device)/1e9,
            calibration_only=True,complete_objective_examples=5000)
    ex.write_json(args.output/'verification.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
