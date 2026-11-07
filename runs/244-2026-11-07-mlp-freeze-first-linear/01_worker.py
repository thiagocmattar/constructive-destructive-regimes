"""Qualified Adam continuation with only first-linear parameter increments suppressed."""
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from intervention import freeze_first_step, cosine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE/'config.json').read_text())
PARENT = ROOT/CFG['parent_run']
SOURCE = ROOT/CFG['parent_worker']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


w = module('native223', PARENT/'01_worker.py')
d = module('capture242', ROOT/'runs/242-2026-11-07-mlp-all-layer-oscillation/diagnostics.py')


def snapshot(c):
    return dict(completed_updates=c['state'],model=w.cpu(c['task'].model.state_dict()),
        optimizer=w.cpu(c['opt'].state_dict()),previous_gradient=w.cpu(c['previous_gradient']),
        current_gradient=w.cpu(c.get('current_gradient')),rng=w.cpu(w.rng()),
        run=CFG['run_id'],config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'))


def save(c,path):
    temp=path.with_suffix('.tmp');torch.save(snapshot(c),temp);temp.replace(path)


def restore(c,p):
    c['task'].model.load_state_dict(p['model']);c['opt'].load_state_dict(p['optimizer'])
    c['state']=p['completed_updates']
    for key in ['previous_gradient','current_gradient']:
        c[key]=None if p[key] is None else p[key].to('cuda:0')
    w.restore_rng(p['rng']);w.validate(c['opt'],c['state'],c['base'])


def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    manifest=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    w.write(out/'manifest.json',manifest)
    c=w.prepare('mlp',json.loads((PARENT/'config.json').read_text()))
    model=c['task'].model;x=c['task'].data['x_train'];input_reference=x.detach().clone()
    reference=w.rows(SOURCE/'prefix/boundaries.jsonl')[:-1]+w.rows(SOURCE/'control/boundaries.jsonl')
    names=['preactivation1','hidden1','preactivation2','hidden2','output']
    all_names=names+['raw_gradient'];count=CFG['dense_capture_end']-CFG['diagnostic_start']+1
    shapes={n:((656810,) if n=='raw_gradient' else (5000,10 if n=='output' else 200)) for n in all_names}
    arrays={n:np.lib.format.open_memmap(out/(n+'.npy'),mode='w+',dtype=np.float32,shape=(count,*shapes[n])) for n in all_names}
    w.write(out/'array_metadata.json',dict(states=list(range(CFG['diagnostic_start'],CFG['dense_capture_end']+1)),
        shapes={n:list(s) for n,s in shapes.items()},dtype='float32',all_examples=5000,
        parameters=[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in model.named_parameters()],
        representation_index='state s; outgoing delta_t=A_(t+1)-A_t',gradient='raw full-objective gradient, not differences'))
    history={};last_loss=None;fixed=None;fixed_rep={};captures=0;freeze_checks=0
    checks=dict(status='running',prefix_pairs=0,qualification_pairs=0,source_capture_states=[],observer_preserves_rng_gradients=True)

    def guard():
        assert not stopped[0] and time.monotonic()-start<CFG['runtime_limit_seconds'] and not (HERE/'STOP').exists()

    def progress(phase,loss):
        elapsed=time.monotonic()-start
        event=dict(phase=phase,state=c['state'],loss=loss,elapsed_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
        if phase=='freeze' and c['state']>CFG['fork']:
            speed=(c['state']-CFG['fork'])/max(time.monotonic()-freeze_start,1e-9)
            event.update(updates_per_second=speed,estimated_remaining_seconds=(CFG['end']-c['state'])/speed)
        w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)

    def boundary(check_reference=False):
        loss,g=c['task'].gradient();c['current_gradient']=g.clone()
        assert np.isfinite(loss) and bool(torch.isfinite(g).all())
        previous=c['previous_gradient'];gn=float(g.double().norm())
        rho=cosine(g,previous) if previous is not None else None
        pn=float(previous.double().norm()) if previous is not None else 0.
        q=1-float(g.double()@previous.double())/pn**2 if pn else None
        active=g[614600:];pa=previous[614600:] if previous is not None else None
        row=dict(state=c['state'],loss=loss,raw_gradient_norm=gn,rho=rho,eta_kappa_hat=q,
            active_gradient_norm=float(active.double().norm()),active_gradient_rho=cosine(active,pa) if pa is not None else None)
        if check_reference:
            ref=reference[c['state']];assert loss==ref['loss'] and gn==ref['raw_gradient_norm'],('reference',c['state'])
            if rho is not None:
                assert abs(rho-ref['rho'])<1e-12 and abs(q-ref['eta_kappa_hat'])<1e-12
        return row,g

    def capture(state,g,qualify=False):
        nonlocal captures
        rng=w.rng();grads=[p.grad.clone() for p in model.parameters()]
        with torch.no_grad():values=d.capture(model,x,CFG['physical_batch'],input_reference)
        assert values[-1]==5000;current=dict(zip(names,values[:-1]));current['raw_gradient']=g
        w.equal(rng,w.rng())
        for p,old in zip(model.parameters(),grads):assert torch.equal(p.grad,old)
        if state==CFG['fork']:
            fixed_rep.update({n:current[n].clone() for n in names[:2]})
        if state>=CFG['fork']:
            for n in names[:2]:assert torch.equal(current[n],fixed_rep[n]),('frozen representation',n,state)
        if qualify:
            sources={
              'preactivation1':('242-2026-11-07-mlp-all-layer-oscillation','preactivation1.npy'),
              'hidden1':('241-2026-11-07-mlp-first-hidden-oscillation','hidden1.npy'),
              'preactivation2':('241-2026-11-07-mlp-first-hidden-oscillation','preactivation2.npy'),
              'hidden2':('240-2026-11-07-mlp-hidden-change-svd','hidden.npy'),
              'raw_gradient':('240-2026-11-07-mlp-hidden-change-svd','gradients.npy')}
            for n,(run,filename) in sources.items():
                original=np.load(ROOT/'runs'/run/'artifacts/attempt-01/worker'/filename,mmap_mode='r')
                assert np.array_equal(current[n].cpu().numpy(),original[state-447]),(n,state)
            prior=torch.load(ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/attempt-01/worker/outputs.pt',map_location='cpu',weights_only=False)
            assert torch.equal(current['output'].cpu(),prior['outputs'][state-447])
            checks['source_capture_states'].append(state)
        if state<=CFG['dense_capture_end']:
            for n in all_names:arrays[n][state-CFG['diagnostic_start']]=current[n].detach().cpu().numpy()
        row={}
        for n in names:
            value=current[n].detach();activation=float(value.double().square().mean().sqrt())
            if n in history:
                old,previous_delta,old_activation=history[n];delta=value.double()-old.double()
                row[n]=dict(change_rms=float(delta.square().mean().sqrt()),
                    rho=cosine(delta,previous_delta) if previous_delta is not None else None,
                    activation_rms_from=old_activation,activation_rms_to=activation)
            else:delta=None
            history[n]=(value.clone(),delta,activation)
        if row:w.append(out/'representations.jsonl',dict(state=state-1,streams=row))
        captures+=1

    try:
        # Reconstruct and qualify the pre-spike fork without changing its policy.
        for state in range(CFG['fork']+1):
            guard();assert c['state']==state
            row,g=boundary(True);checks['prefix_pairs']+=1
            row['loss_change']=None if last_loss is None else row['loss']-last_loss;last_loss=row['loss']
            w.append(out/'boundaries.jsonl',row)
            if state>=CFG['diagnostic_start']:capture(state,g,qualify=True)
            if state==CFG['fork']:break
            c['opt'].step();c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']%64==0:progress('prefix',row['loss'])
        save(c,out/'fork.pt');fork=snapshot(c)
        fixed={n:p.detach().clone() for n,p in model[1].named_parameters()}
        assert sum(p.numel() for p in model[1].parameters())==614600
        for state in range(CFG['fork'],CFG['qualification_end']+1):
            guard();assert c['state']==state
            row,g=boundary(True);checks['qualification_pairs']+=1;w.append(out/'qualification_boundaries.jsonl',row)
            if state==CFG['qualification_end']:
                saved=torch.load(SOURCE/'prefix/end.pt',map_location='cpu',weights_only=False)
                for key,value in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(value),saved[key])
                save(c,out/'qualification512.pt');checks['full_state512_exact']=True;break
            c['opt'].step();c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']==CFG['fork']+1:native_first=snapshot(c)
        restore(c,fork);freeze_start=time.monotonic()
        checks['status']='passed';w.write(out/'qualification.json',checks)
        previous_update=None
        for state in range(CFG['fork'],CFG['end']+1):
            guard();assert state==c['state'];row,g=boundary(False)
            if state==CFG['fork']:
                assert torch.equal(g.cpu(),fork['current_gradient'])
            else:
                row['loss_change']=row['loss']-last_loss;w.append(out/'boundaries.jsonl',row);capture(state,g)
            last_loss=row['loss'];w.validate(c['opt'],state,c['base'])
            for n,p in model[1].named_parameters():assert torch.equal(p,fixed[n])
            freeze_checks+=1
            if state==CFG['end']:break
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            freeze_first_step(model,c['opt'],fixed)
            delta=torch.cat([(p.detach().double()-before[n].double()).flatten() for n,p in model.named_parameters()])
            assert float(delta[:614600].norm())==0
            per_layer={}
            for layer in [1,3,5]:
                dp=torch.cat([(p.detach().double()-before[f'{layer}.{n}'].double()).flatten() for n,p in model[layer].named_parameters()])
                per_layer[str(layer)]=dict(actual_increment_norm=float(dp.norm()),parameter_norm=float(torch.cat([p.detach().double().flatten() for p in model[layer].parameters()]).norm()))
            w.append(out/'updates.jsonl',dict(state=state,actual_increment_norm=float(delta.norm()),
                actual_increment_rho=cosine(delta,previous_update) if previous_update is not None else None,
                gradient_dot_actual_step=float(g.double()@delta),layers=per_layer,lr=c['base']))
            previous_update=delta;c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']=state+1
            if c['state']==CFG['fork']+1:
                for n,p in model.named_parameters():
                    if not n.startswith('1.'):w.equal(p.cpu(),native_first['model'][n])
                w.equal(w.cpu(c['opt'].state_dict()),native_first['optimizer']);w.equal(w.cpu(w.rng()),native_first['rng'])
                checks['first_update_later_parameters_and_all_adam_states_exact']=True;w.write(out/'qualification.json',checks)
            if c['state']%CFG['checkpoint_every']==0:
                save(c,out/'recovery.pt')
                for a in arrays.values():a.flush()
            if c['state']%32==0:progress('freeze',row['loss'])
        save(c,out/'end.pt')
        for a in arrays.values():a.flush()
        w.write(out/'freeze_verification.json',dict(status='passed',parameter_boundaries=freeze_checks,
            fixed_first_weight_and_bias=True,first_Z1_H1_exactly_constant_after_fork=True,
            captures=captures,all_inputs_exact=True,all_adam_clocks_native=True,first_layer_parameters=614600,total_parameters=656810))
        manifest.update(status='completed',completed_state=c['state'],fork=CFG['fork'],intervention_updates=CFG['end']-CFG['fork'],
            dense_states=count,captured_states=captures,peak_allocated_bytes=torch.cuda.max_memory_allocated(),freeze_elapsed_seconds=time.monotonic()-freeze_start)
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
