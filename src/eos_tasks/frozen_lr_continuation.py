"""Qualified frozen-first-layer continuation with a configured LR change at500."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks.frozen_first_linear import freeze_first_step,cosine



def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m





def snapshot(c):
    return dict(run=CFG['run_id'],config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'),
        completed_updates=c['state'],model=w.cpu(c['task'].model.state_dict()),optimizer=w.cpu(c['opt'].state_dict()),
        previous_gradient=w.cpu(c['previous_gradient']),current_gradient=w.cpu(c.get('current_gradient')),rng=w.cpu(w.rng()))


def save(c,path):
    temp=path.with_suffix('.tmp');torch.save(snapshot(c),temp);temp.replace(path)


def restore(c,p):
    c['task'].model.load_state_dict(p['model']);c['opt'].load_state_dict(p['optimizer']);c['state']=p['completed_updates']
    for k in ['previous_gradient','current_gradient']:c[k]=None if p[k] is None else p[k].to('cuda:0')
    w.restore_rng(p['rng'])


def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker';out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic();stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    manifest=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    w.write(out/'manifest.json',manifest)
    c=w.prepare('mlp',json.loads((ROOT/CFG['native_run']/'config.json').read_text()))
    parent=torch.load(SOURCE/'fork.pt',map_location='cpu',weights_only=False);restore(c,parent)
    assert c['state']==448
    model=c['task'].model;x=c['task'].data['x_train'];input_reference=x.detach().clone()
    fixed={n:p.detach().clone() for n,p in model[1].named_parameters()}
    names=['preactivation1','hidden1','preactivation2','hidden2','output'];all_names=names+['raw_gradient']
    prior={n:np.load(SOURCE/(n+'.npy'),mmap_mode='r') for n in all_names}
    refs={q['state']:q for q in w.rows(SOURCE/'boundaries.jsonl')}
    count=CFG['dense_capture_end']-CFG['diagnostic_start']+1
    shapes={n:((656810,) if n=='raw_gradient' else (5000,10 if n=='output' else 200)) for n in all_names}
    arrays={n:np.lib.format.open_memmap(out/(n+'.npy'),mode='w+',dtype=np.float32,shape=(count,*shapes[n])) for n in all_names}
    w.write(out/'array_metadata.json',dict(states=list(range(CFG['diagnostic_start'],CFG['dense_capture_end']+1)),
        shapes={n:list(s) for n,s in shapes.items()},dtype='float32',examples=5000,raw_gradient='all656810 parameters before Adam',
        difference_index='outgoing delta_t=A_(t+1)-A_t; state500 is matched before changed update500->501'))
    checks=dict(status='running',exact_boundary_states=[],exact_raw_gradient_states=[],exact_representation_states=[])
    captures={};fixed_rep={};frozen_checks=0

    def guard():
        assert not stopped[0] and time.monotonic()-begin<CFG['runtime_limit_seconds'] and not (HERE/'STOP').exists()

    def boundary():
        loss,g=c['task'].gradient();c['current_gradient']=g.clone();previous=c['previous_gradient']
        assert np.isfinite(loss) and bool(torch.isfinite(g).all())
        pn=float(previous.double().norm());q=1-float(g.double()@previous.double())/pn**2 if pn else None
        return dict(state=c['state'],loss=loss,raw_gradient_norm=float(g.double().norm()),rho=cosine(g,previous),eta_kappa_hat=q,
            active_gradient_norm=float(g[614600:].double().norm()),active_gradient_rho=cosine(g[614600:],previous[614600:]),
            applied_lr=None if c['state']==CFG['end'] else c['opt'].param_groups[0]['lr']),g

    def capture(g):
        rng=w.rng();grads=[p.grad.clone() for p in model.parameters()]
        with torch.no_grad():values=d.capture(model,x,CFG['physical_batch'],input_reference)
        assert values[-1]==5000;current=dict(zip(names,values[:-1]));current['raw_gradient']=g
        w.equal(rng,w.rng())
        for p,old in zip(model.parameters(),grads):assert torch.equal(p.grad,old)
        if not fixed_rep:fixed_rep.update({n:current[n].clone() for n in names[:2]})
        for n in names[:2]:assert torch.equal(current[n],fixed_rep[n])
        return current

    def persist(current,state):
        for n in all_names:arrays[n][state-CFG['diagnostic_start']]=current[n].detach().cpu().numpy()

    try:
        # The retained complete448 state and every captured gradient qualify the
        # short reconstruction; no historical complete500 checkpoint existed.
        for state in range(448,502):
            guard();assert c['state']==state;w.validate(c['opt'],state,CFG['base_lr'])
            row,g=boundary();ref=refs[state]
            for k in ['loss','raw_gradient_norm']:assert row[k]==ref[k],(state,k)
            for k in ['rho','eta_kappa_hat','active_gradient_norm','active_gradient_rho']:
                assert abs(row[k]-ref[k])<1e-12,(state,k)
            assert np.array_equal(g.cpu().numpy(),prior['raw_gradient'][state-447])
            checks['exact_boundary_states'].append(state);checks['exact_raw_gradient_states'].append(state)
            w.append(out/'qualification_boundaries.jsonl',row)
            if state>=499:
                current=capture(g)
                for n in names:assert np.array_equal(current[n].cpu().numpy(),prior[n][state-447]),(n,state)
                checks['exact_representation_states'].append(state)
                if state<=500:captures[state]={n:current[n].clone() for n in names};persist(current,state)
            if state==499:params499=w.flat(model.parameters()).detach().double().clone()
            if state==500:
                fork=snapshot(c);save(c,out/'fork500.pt');params500=w.flat(model.parameters()).detach().double().clone()
            if state==501:
                native501=snapshot(c);save(c,out/'qualification501.pt');break
            freeze_first_step(model,c['opt'],fixed);c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
        restore(c,fork)
        for group in c['opt'].param_groups:group['lr']=CFG['intervention_lr']
        save(c,out/'intervention_start500.pt')
        history={n:(captures[500][n],captures[500][n].double()-captures[499][n].double()) for n in names}
        previous_update=params500-params499;last_loss=refs[499]['loss'];intervention_start=time.monotonic()
        checks.update(status='passed',observer_preserves_rng_gradients=True,fork_restored_exactly=True)
        w.write(out/'qualification.json',checks)
        for state in range(500,CFG['end']+1):
            guard();assert c['state']==state;w.validate(c['opt'],state,CFG['intervention_lr'])
            row,g=boundary();row['loss_change']=row['loss']-last_loss;last_loss=row['loss'];w.append(out/'boundaries.jsonl',row)
            if state==500:assert torch.equal(g.cpu(),fork['current_gradient']) and row['loss']==refs[500]['loss']
            else:
                current=capture(g)
                if state<=CFG['dense_capture_end']:persist(current,state)
                rep={}
                for n in names:
                    old,old_delta=history[n];value=current[n].detach();delta=value.double()-old.double()
                    rep[n]=dict(rho=cosine(delta,old_delta),change_rms=float(delta.square().mean().sqrt()),
                        activation_rms_from=float(old.double().square().mean().sqrt()),activation_rms_to=float(value.double().square().mean().sqrt()))
                    history[n]=(value.clone(),delta)
                w.append(out/'representations.jsonl',dict(state=state-1,streams=rep))
            for n,p in model[1].named_parameters():assert torch.equal(p,fixed[n])
            frozen_checks+=1
            if state==CFG['end']:break
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            freeze_first_step(model,c['opt'],fixed)
            delta=torch.cat([(p.detach().double()-before[n].double()).flatten() for n,p in model.named_parameters()])
            assert float(delta[:614600].norm())==0
            layers={}
            for layer in [1,3,5]:
                dp=torch.cat([(p.detach().double()-before[f'{layer}.{n}'].double()).flatten() for n,p in model[layer].named_parameters()])
                layers[str(layer)]=dict(actual_increment_norm=float(dp.norm()))
            w.append(out/'updates.jsonl',dict(state=state,actual_increment_norm=float(delta.norm()),actual_increment_rho=cosine(delta,previous_update),
                gradient_dot_actual_step=float(g.double()@delta),layers=layers,lr=CFG['intervention_lr']))
            previous_update=delta;c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']==501:
                w.equal(w.cpu(c['opt'].state_dict())['state'],native501['optimizer']['state'])
                w.equal(w.cpu(w.rng()),native501['rng'])
                errors=[]
                for n,p in model.named_parameters():
                    if not n.startswith('1.'):
                        factor=CFG['intervention_lr']/CFG['base_lr']
                        expected=fork['model'][n].double()+factor*(native501['model'][n].double()-fork['model'][n].double())
                        errors.append(float((p.detach().cpu().double()-expected).abs().max()))
                        torch.testing.assert_close(p.detach().cpu().double(),expected,rtol=0,atol=max(2e-7,factor*1e-7))
                checks.update(first_changed_step_moments_and_clocks_exact=True,first_changed_step_scaled_increment_max_abs_error=max(errors),first_step_lr_ratio=factor)
                w.write(out/'qualification.json',checks)
            if c['state']%CFG['checkpoint_every']==0:
                save(c,out/'recovery.pt')
                for a in arrays.values():a.flush()
            if c['state']%32==0:
                elapsed=time.monotonic()-intervention_start;speed=(c['state']-500)/elapsed
                event=dict(state=c['state'],loss=row['loss'],loss_state=state,elapsed_seconds=time.monotonic()-begin,
                    updates_per_second=speed,estimated_remaining_seconds=(CFG['end']-c['state'])/speed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save(c,out/'end.pt')
        for a in arrays.values():a.flush()
        w.write(out/'freeze_verification.json',dict(status='passed',parameter_boundaries=frozen_checks,
            fixed_first_weight_and_bias=True,first_Z1_H1_exactly_constant=True,all_inputs_exact=True,all_adam_clocks_native=True))
        manifest.update(status='completed',completed_state=c['state'],intervention_updates=301,dense_states=count,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),intervention_elapsed_seconds=time.monotonic()-intervention_start)
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-begin)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


def run(here):
    global HERE,CFG,SOURCE,w,d
    HERE=Path(here);CFG=json.loads((HERE/'config.json').read_text());SOURCE=ROOT/CFG['parent_worker']
    w=module('native223',ROOT/CFG['native_run']/'01_worker.py')
    d=module('capture242',ROOT/'runs/242-2026-11-07-mlp-all-layer-oscillation/diagnostics.py')
    main()
