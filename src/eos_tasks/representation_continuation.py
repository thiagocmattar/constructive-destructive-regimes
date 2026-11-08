"""Matched full-objective continuations for the three fixed EOS architectures."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch

from src.cifar_eos.core import flat
from src.eos_tasks import representation_policy as policy

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('parent223',ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def snapshot(c):
    return dict(model=w.cpu(c['task'].model.state_dict()),optimizer=w.cpu(c['opt'].state_dict()),
        param_group_names=[[n for n,p in c['task'].model.named_parameters() if any(p is q for q in group['params'])] for group in c['opt'].param_groups],
        previous_gradient=w.cpu(c['previous_gradient']),current_gradient=w.cpu(c.get('current_gradient')),
        rng=w.cpu(w.rng()),state=c['state'])


def restore(c, saved):
    model=c['task'].model;model.load_state_dict(saved['model']);params=dict(model.named_parameters())
    groups=[dict(params=[params[n] for n in names],lr=c['base']) for names in saved['param_group_names']]
    c['opt']=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
    c['opt'].load_state_dict(copy.deepcopy(saved['optimizer']))
    c['state']=saved['state']
    for k in ['previous_gradient','current_gradient']:
        c[k]=None if saved[k] is None else saved[k].to('cuda').clone()
    w.restore_rng(saved['rng'])


def save(c,path):
    temporary=path.with_suffix('.tmp');torch.save(snapshot(c),temporary);os.replace(temporary,path)


def fingerprint(saved):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(saved);return h.hexdigest()


def native_update(c):
    loss,g=c['task'].gradient();c['current_gradient']=g.clone()
    c['opt'].step();c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
    return loss,g


def prepare(run,cfg,out):
    c=w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
    if cfg['case']!='mlp':w.load(c,run/'inputs/prefix.pt')
    assert c['state']==cfg['source_state'];c['fork']=cfg['fork']
    model=c['task'].model;groups=policy.group_layout(model);entry,trunk=policy.affine_targets(model,cfg['case'])
    ema={};pools={n:dict(coordinate=[],functional=[]) for n in trunk};history=[];probe_history={};parameter_history=[]
    cache=out/'shared';cache.mkdir(exist_ok=False)
    start=time.monotonic();sample_begin=cfg['fork']-cfg['calibration_updates']
    reference={q['state']:q for q in w.rows(run/'inputs/boundaries.jsonl')} if cfg['case']!='mlp' else {}
    while c['state']<cfg['fork']:
        t=c['state'];loss,g=c['task'].gradient();c['current_gradient']=g.clone()
        capture=t>=sample_begin
        metrics=policy.block_diagnostics(g,c['previous_gradient'],groups,ema) if capture else None
        if capture:
            rng=w.rng();values,inputs,probe_loss=policy.capture(c['task'],cfg['case'],cfg['probe'],trunk);w.equal(rng,w.rng())
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            parameter_history=(parameter_history+[flat(before.values()).detach().cpu()])[-2:]
            for n,v in values.items():
                old,_=probe_history.get(n,(None,None))
                probe_history[n]=(v,None if old is None else v.double()-old.double())
        c['opt'].step()
        if capture:
            for n in trunk:
                module,weight,bias=policy.affine_parameters(model,n)
                wn=n+('.in_proj_weight' if isinstance(module,torch.nn.MultiheadAttention) else '.weight')
                bn=n+('.in_proj_bias' if isinstance(module,torch.nn.MultiheadAttention) else '.bias')
                dw=weight.detach().double()-before[wn].double();db=None if bias is None else bias.detach().double()-before[bn].double()
                pools[n]['coordinate'].append(torch.cat([dw.flatten()]+([] if db is None else [db.flatten()])).abs().cpu().numpy())
                pools[n]['functional'].append(policy.affine_scores(module,inputs[n],dw,db).cpu().numpy())
            history.append(dict(state=t,loss=loss,raw_gradient_norm=float(g.double().norm()),blocks=metrics,
                historical_loss=reference.get(t,{}).get('loss'),probe_loss=probe_loss))
            policy.advance_ema(ema,metrics)
        c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
        if capture or t%64==0:w.write(out/'progress.json',dict(phase='prefix',state=c['state'],loss=loss,elapsed_seconds=time.monotonic()-start))
    thresholds={};arrays={}
    for i,(n,values) in enumerate(pools.items()):
        thresholds[n]={}
        for kind,data in values.items():
            a=np.stack(data);arrays[f'{i}_{kind}']=a
            thresholds[n][kind]={str(q):float(np.quantile(a,q,method='linear')) for q in cfg['quantiles']}
    np.savez(cache/'calibration.npz',**arrays)
    probe_history=w.cpu(probe_history)
    torch.save(dict(parameter_history=parameter_history,states=[cfg['fork']-2,cfg['fork']-1],probe_history_fingerprint=fingerprint(probe_history)),cache/'prefix_audit.pt')
    w.write(cache/'thresholds.json',dict(modules=thresholds,module_order=trunk,window=[sample_begin,cfg['fork']-1],ema=ema,history=history))
    loss,g=c['task'].gradient();c['current_gradient']=g.clone();save(c,cache/'fork.pt');fork=snapshot(c)
    native_update(c);native_update(c);direct=snapshot(c)
    restore(c,fork);native_update(c);one=snapshot(c);restore(c,one);native_update(c)
    w.equal(direct,snapshot(c));restore(c,fork)
    w.write(cache/'qualification.json',dict(status='passed',direct2_vs_resume_exact=True,fork_fingerprint=fingerprint(fork),
        native_two_step_fingerprint=fingerprint(direct),fork_loss=loss,source_state=cfg['source_state'],fork_state=cfg['fork'],
        historical_comparison=history,elapsed_seconds=time.monotonic()-start))
    if cfg['case']=='mlp' and (run/'inputs/reference448.pt').exists():
        old=torch.load(run/'inputs/reference448.pt',map_location='cpu',weights_only=False)
        for k in ['model','optimizer','rng','previous_gradient','current_gradient']:w.equal(fork[k],old[k])
    del direct,one
    return c,fork,thresholds,ema,groups,entry,trunk,probe_history,parameter_history


def run_arm(c,fork,cfg,arm,folder,thresholds,initial_ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
    folder.mkdir(exist_ok=False);restore(c,fork);model=c['task'].model;begin=time.monotonic()
    end=cfg['fork']+3 if mechanics else cfg['end']+1
    history={n:(v.to('cuda').clone(),d.to('cuda').clone()) for n,(v,d) in probe_history.items()}
    prior_parameters=[v.to('cuda').clone() for v in parameter_history];ema=copy.deepcopy(initial_ema)
    multiplier=arm.get('embedding_lr_multiplier',1.)
    if multiplier!=1:
        assert cfg['case']=='transformer';embedding=model.encoder.weight
        original=c['opt'].param_groups[0];others=[p for p in original['params'] if p is not embedding]
        c['opt'].param_groups=[dict(original,params=others),dict(original,params=[embedding],lr=c['base']*multiplier)]
    targets=entry if arm.get('scope')=='entry' else trunk
    needed=targets if arm['kind']=='functional' else []
    status=dict(status='running',arm=arm,case=cfg['case'],started_utc=w.stamp(),fork=cfg['fork'],last_loss_state=cfg['end'],
        mechanics_only=mechanics,environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
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
            assert np.isfinite(loss) and bool(torch.isfinite(g).all())
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
            if state==cfg['end']:
                evaluation={name:c['task'].evaluate(name) for name in ['valid','test'] if 'x_'+name in c['task'].data}
                w.write(folder/'evaluation.json',dict(state=state,splits=evaluation,selection='Secondary only; policies fixed before held-out evaluation'))
            if state==end:break
            before={n:p.detach().clone() for n,p in model.named_parameters()};before_flat=flat(before.values()).double()
            c['opt'].step();proposed=flat(model.parameters()).detach().double()-before_flat
            native_state=fingerprint({n:w.cpu(c['opt'].state[p]) for n,p in model.named_parameters()}) if state==cfg['fork'] else None
            gates=policy.apply_policy(model,before,arm,groups,metrics,inputs,thresholds,targets)
            actual=flat(model.parameters()).detach().double()-before_flat
            assert bool(torch.isfinite(actual).all())
            if native_state is not None:assert native_state==fingerprint({n:w.cpu(c['opt'].state[p]) for n,p in model.named_parameters()})
            gn=metrics['__global__']['norm'];an=float(actual.norm());pn=float(proposed.norm())
            update=dict(state=state,proposed_norm=pn,actual_norm=an,retained_norm_ratio=an/pn if pn else None,
                raw_gradient_dot_actual=float(g.double()@actual),raw_gradient_actual_cosine=policy.cosine(g,actual),
                actual_update_rho=policy.cosine(actual,previous_actual),gates=gates,
                embedding_lr=c['base']*multiplier if cfg['case']=='transformer' else None,
                all_optimizer_steps=int(next(iter(c['opt'].state.values()))['step']),seconds=time.monotonic()-tick)
            w.append(folder/'updates.jsonl',update)
            if state in audit_states:
                torch.save(dict(state=state,parameter_names=[(n,list(p.shape)) for n,p in model.named_parameters()],
                    before_parameters=before_flat.float().cpu(),previous_parameters=[p.cpu() for p in prior_parameters],
                    raw_gradient=g.cpu(),previous_gradient=None if c['previous_gradient'] is None else c['previous_gradient'].cpu(),
                    proposed_parameters=(before_flat+proposed).float().cpu(),actual_parameters=(before_flat+actual).float().cpu(),
                    blocks=metrics,gates=gates),folder/f'audit_{state}.pt')
            prior_parameters=(prior_parameters+[before_flat.float()])[-2:]
            previous_actual=actual;c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
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
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arms',default='');parser.add_argument('--mechanics',action='store_true');args=parser.parse_args()
    run=args.run.resolve();cfg=json.loads((run/'config.json').read_text());out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for item in json.loads((run/'input_manifest.json').read_text())['files']:
        f=ROOT/item['path'];assert f.stat().st_size==item['bytes'] and w.sha(f)==item['sha256'],str(f)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    c,fork,thresholds,ema,groups,entry,trunk,probe_history,parameter_history=prepare(run,cfg,out)
    w.write(out/'layout.json',dict(groups=groups,entry=entry,trunk=trunk,probe=cfg['probe'],case=cfg['case']))
    selected=args.arms.split(',') if args.arms else [a['id'] for a in cfg['arms']]
    assert selected[0]=='control' and set(selected)<=set(a['id'] for a in cfg['arms'])
    statuses=[]
    for name in selected:
        arm=next(a for a in cfg['arms'] if a['id']==name)
        statuses.append(run_arm(c,fork,cfg,arm,out/name,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,args.mechanics))
    w.write(out/'manifest.json',dict(status='completed',case=cfg['case'],arms=selected,results=statuses,mechanics_only=args.mechanics))


if __name__=='__main__':main()
