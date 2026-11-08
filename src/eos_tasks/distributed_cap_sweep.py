"""H049: MLP-only distributed local cap x global LR, with fixed 1x budgets.

The arm loop is derived from representation_continuation.run_arm. Its frozen
H048 source remains untouched. Additions: global LR at branch start, periodic
complete held-out evaluation and native-Adam optimizer audits. Prefix calibration,
probe capture and the per-unit cap reuse the already verified implementation.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from src.cifar_eos.core import flat
from src.eos_tasks import representation_continuation as parent
from src.eos_tasks import representation_policy as policy

ROOT=parent.ROOT
w=parent.w
restore=parent.restore
save=parent.save
fingerprint=parent.fingerprint


def prepare_seed(run,cfg,out):
    original=parent.w.prepare
    def seeded(case,parent_cfg):
        context=original(case,parent_cfg)
        seed=cfg['seed']
        if seed:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        state=torch.load(ROOT/cfg['initial_state'],map_location='cpu',weights_only=True)
        context['task'].model.load_state_dict(state)
        assert w.tensor_hash(context['task'].model.parameters())==cfg['initial_tensor_sha256']
        assert not context['opt'].state and context['state']==0
        assert context['base']==cfg['calibration_lr']==cfg['base_lr']==.001
        return context
    parent.w.prepare=seeded
    try: return parent.prepare(run,cfg,out)
    finally: parent.w.prepare=original


def run_arm(c,fork,cfg,arm,folder,thresholds,initial_ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
    folder.mkdir(exist_ok=False);restore(c,fork);model=c['task'].model;begin=time.monotonic()
    end=cfg['fork']+3 if mechanics else cfg['end']+1
    history={n:(v.to('cuda').clone(),d.to('cuda').clone()) for n,(v,d) in probe_history.items()}
    prior_parameters=[v.to('cuda').clone() for v in parameter_history];ema=copy.deepcopy(initial_ema)
    assert cfg['case']=='mlp'
    multiplier=arm['lr_multiplier']; applied_lr=cfg['base_lr']*multiplier
    for group in c['opt'].param_groups: group['lr']=applied_lr
    assert all(int(s['step'])==cfg['fork'] for s in c['opt'].state.values())
    targets=trunk
    needed=targets if arm['kind']=='functional' else []
    status=dict(status='running',arm=arm,case=cfg['case'],started_utc=w.stamp(),fork=cfg['fork'],last_loss_state=cfg['end'],
        mechanics_only=mechanics,applied_lr=applied_lr,calibration_lr=cfg['calibration_lr'],environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        training_units=c['task'].units,probe_units=cfg['probe'],parameter_count=sum(p.numel() for p in model.parameters()))
    w.write(folder/'manifest.json',status)
    stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    audit_states={cfg['fork'],cfg['fork']+32,cfg['end']}
    previous_actual=None
    try:
        while c['state']<=end:
            assert time.monotonic()-begin<cfg['seconds_per_arm'] and not stop[0] and not (folder.parent/'STOP').exists()
            state=c['state'];tick=time.monotonic();loss,g=c['task'].gradient();c['current_gradient']=g.clone()
            if not np.isfinite(loss) or not bool(torch.isfinite(g).all()): raise FloatingPointError('Nonfinite training loss or gradient')
            metrics=policy.block_diagnostics(g,c['previous_gradient'],groups,ema)
            rng=w.rng();values,inputs,probe_loss=policy.capture(c['task'],cfg['case'],cfg['probe'],needed);w.equal(rng,w.rng())
            # The observer does not touch .grad; the flattened raw gradient is checked.
            assert torch.equal(flat([p.grad for p in model.parameters()]),g)
            representations={}
            for n,v in values.items():
                old,previous_delta=history.get(n,(None,None));delta=None if old is None else v.double()-old.double()
                representations[n]=dict(activation_rms=float(v.double().square().mean().sqrt()),
                    change_rms=None if delta is None else float(delta.square().mean().sqrt()),rho=None if delta is None else policy.cosine(delta,previous_delta),
                    coordinates=v.numel())
                history[n]=(v,delta)
            # Row state s describes incoming changes s-1 -> s. The reducer shifts
            # those to outgoing t=s-1. Two untreated predecessor states are kept.
            row=dict(state=state,loss=loss,probe_loss=probe_loss,raw_gradient_norm=metrics['__global__']['norm'],
                rho=metrics['__global__']['rho'],eta_kappa_hat=metrics['__global__']['eta_kappa_hat'],
                blocks=metrics,representations=representations,elapsed_seconds=time.monotonic()-begin)
            w.append(folder/'boundaries.jsonl',row)
            if state<=cfg['end'] and ((state-cfg['fork'])%cfg['heldout_every']==0 or state==cfg['end']):
                eval_rng=w.rng()
                evaluation={name:c['task'].evaluate(name) for name in ['valid','test'] if 'x_'+name in c['task'].data}
                w.equal(eval_rng,w.rng()); assert torch.equal(flat([p.grad for p in model.parameters()]),g)
                w.append(folder/'evaluations.jsonl',dict(state=state,splits=evaluation,selection='Prespecified descriptive evaluation; never tunes or stops training'))
            if state==end:break
            before={n:p.detach().clone() for n,p in model.named_parameters()};before_flat=flat(before.values()).double()
            optimizer_before=w.cpu(c['opt'].state_dict()) if state in audit_states else None
            assert all(group['lr']==applied_lr for group in c['opt'].param_groups)
            c['opt'].step();proposed=flat(model.parameters()).detach().double()-before_flat
            native_state=fingerprint({n:w.cpu(c['opt'].state[p]) for n,p in model.named_parameters()}) if state in audit_states else None
            gates=policy.apply_policy(model,before,arm,groups,metrics,inputs,thresholds,targets)
            actual=flat(model.parameters()).detach().double()-before_flat
            if not bool(torch.isfinite(actual).all()): raise FloatingPointError('Nonfinite applied parameter increment')
            if native_state is not None:assert native_state==fingerprint({n:w.cpu(c['opt'].state[p]) for n,p in model.named_parameters()})
            gn=metrics['__global__']['norm'];an=float(actual.norm());pn=float(proposed.norm())
            update=dict(state=state,proposed_norm=pn,actual_norm=an,retained_norm_ratio=an/pn if pn else None,
                raw_gradient_dot_actual=float(g.double()@actual),raw_gradient_actual_cosine=policy.cosine(g,actual),
                actual_update_rho=policy.cosine(actual,previous_actual),gates=gates,
                applied_lr=applied_lr,lr_multiplier=multiplier,calibration_lr=cfg['calibration_lr'],
                all_optimizer_steps=int(next(iter(c['opt'].state.values()))['step']),seconds=time.monotonic()-tick)
            w.append(folder/'updates.jsonl',update)
            if state in audit_states:
                torch.save(dict(state=state,parameter_names=[(n,list(p.shape)) for n,p in model.named_parameters()],
                    before_parameters=before_flat.float().cpu(),previous_parameters=[p.cpu() for p in prior_parameters],
                    raw_gradient=g.cpu(),previous_gradient=None if c['previous_gradient'] is None else c['previous_gradient'].cpu(),
                    proposed_parameters=(before_flat+proposed).float().cpu(),actual_parameters=(before_flat+actual).float().cpu(),
                    blocks=metrics,gates=gates,optimizer_before=optimizer_before,
                    optimizer_after=w.cpu(c['opt'].state_dict())),folder/f'audit_{state}.pt')
            prior_parameters=(prior_parameters+[before_flat.float()])[-2:]
            previous_actual=actual;c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            assert all(int(s['step'])==c['state'] for s in c['opt'].state.values())
            policy.advance_ema(ema,metrics)
            if c['state']%cfg['checkpoint_every']==0:save(c,folder/'recovery.pt')
            progress=dict(phase='arm',case=cfg['case'],arm=arm['id'],state=c['state'],loss=loss,loss_state=state,
                elapsed_seconds=time.monotonic()-begin,updates_per_second=(c['state']-cfg['fork'])/(time.monotonic()-begin),
                peak_cuda_bytes=torch.cuda.max_memory_allocated())
            w.write(folder/'progress.json',progress)
            if c['state']%16==0:print(json.dumps(progress),flush=True)
        save(c,folder/'end.pt')
        status.update(status='completed',completed_state=c['state'],updates=c['state']-cfg['fork'],peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        save(c,folder/'recovery.pt');status.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-begin,finished_utc=w.stamp());w.write(folder/'manifest.json',status)
    return status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mechanics',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve();cfg=json.loads((run/'config.json').read_text())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for item in json.loads((run/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    assert cfg['case']=='mlp' and cfg['seed'] in [0,1,2]
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    started=time.monotonic()
    c,fork,thresholds,ema,groups,entry,trunk,probe_history,parameter_history=prepare_seed(run,cfg,out)
    w.write(out/'layout.json',dict(groups=groups,entry=entry,trunk=trunk,probe=cfg['probe'],case=cfg['case']))
    frozen_thresholds=json.dumps(thresholds,sort_keys=True)
    results=[]
    for i,arm in enumerate(cfg['arms']):
        w.write(out/'progress.json',dict(phase='arm',arm=arm['id'],completed_arms=i,total_arms=len(cfg['arms']),elapsed_seconds=time.monotonic()-started))
        try:
            result=run_arm(c,fork,cfg,arm,out/arm['id'],thresholds,ema,groups,entry,trunk,probe_history,parameter_history,args.mechanics)
        except FloatingPointError:
            result=json.loads((out/arm['id']/'manifest.json').read_text())
        assert json.dumps(thresholds,sort_keys=True)==frozen_thresholds
        results.append(result)
        w.write(out/'manifest.json',dict(status='running',case=cfg['case'],seed=cfg['seed'],results=results,mechanics_only=args.mechanics))
    w.write(out/'manifest.json',dict(status='completed',case=cfg['case'],seed=cfg['seed'],results=results,
        mechanics_only=args.mechanics,elapsed_seconds=time.monotonic()-started,finished_utc=w.stamp()))


if __name__=='__main__': main()
