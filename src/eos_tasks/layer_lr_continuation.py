"""Warm MLP Adam continuation with per-layer LR and unchanged optimizer state."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from src.eos_tasks.frozen_first_linear import cosine


def layer_step(model, optimizer, first_lr, later_lr):
    """Use native Adam per parameter; restore the checkpoint's group layout."""
    original = optimizer.param_groups
    assert len(original) == 1
    ordered = [p for layer in [1, 3, 5] for p in model[layer].parameters()]
    assert all(a is b for a, b in zip(ordered, original[0]['params']))
    assert len(ordered) == len(original[0]['params'])
    optimizer.param_groups = [dict(original[0], params=list(model[layer].parameters()),
                                  lr=first_lr if layer == 1 else later_lr) for layer in [1, 3, 5]]
    try:
        optimizer.step()
    finally:
        optimizer.param_groups = original


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main(here):
    here = Path(here).resolve(); root = here.parents[1]
    cfg = json.loads((here/'config.json').read_text())
    w = module('native223', root/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
    d = module('capture242', root/'runs/242-2026-11-07-mlp-all-layer-oscillation/diagnostics.py')
    source = root/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/attempt-01/worker'
    historical = root/'runs/223-2026-11-04-multispike-interventions/artifacts/worker'
    for item in json.loads((here/'input_manifest.json').read_text())['files']:
        p = root/item['path']
        assert p.stat().st_size == item['bytes'] and w.sha(p) == item['sha256'], str(p)
    w.configure(); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    attempt = os.environ.get('MLP_LR_ATTEMPT', 'attempt-01')
    assert attempt in ['attempt-01','attempt-02']
    out = here/'artifacts'/attempt/'worker'; out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); stopped = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    manifest = dict(status='running', started_utc=w.stamp(), gpu=torch.cuda.get_device_name(),
                    torch=torch.__version__, cuda=torch.version.cuda)
    w.write(out/'manifest.json', manifest)
    c = w.prepare('mlp', json.loads((root/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
    model = c['task'].model; x = c['task'].data['x_train']; input_reference = x.clone()
    parent = torch.load(source/'fork.pt', map_location='cpu', weights_only=False)
    refs = {q['state']: q for q in w.rows(historical/'prefix/boundaries.jsonl')[:-1] + w.rows(historical/'control/boundaries.jsonl')}
    names = ['preactivation1', 'hidden1', 'preactivation2', 'hidden2', 'output']
    all_names = names + ['raw_gradient']
    prior = {n: np.load(source/(n+'.npy'), mmap_mode='r') for n in all_names}
    arrays = {n: np.lib.format.open_memmap(out/(n+'.npy'), mode='w+', dtype=np.float32,
                shape=(55, *prior[n].shape[1:])) for n in all_names}
    for n in all_names: arrays[n][0] = prior[n][0]  # untreated state447
    w.write(out/'array_metadata.json', dict(states=list(range(447,502)), all_examples=5000,
        dtype='float32', shapes={n:list(a.shape) for n,a in arrays.items()},
        parameters=[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in model.named_parameters()],
        representation_index='outgoing delta_t=A_(t+1)-A_t', gradient='raw, not differences'))

    def restore(p):
        model.load_state_dict(p['model']); c['opt'].load_state_dict(copy.deepcopy(p['optimizer']))
        c['state'] = p['completed_updates']
        for key in ['previous_gradient','current_gradient']:
            c[key] = None if p[key] is None else p[key].to('cuda:0')
        w.restore_rng(p['rng'])

    def snapshot():
        return dict(completed_updates=c['state'], model=w.cpu(model.state_dict()),
            optimizer=w.cpu(c['opt'].state_dict()), previous_gradient=w.cpu(c['previous_gradient']),
            current_gradient=w.cpu(c.get('current_gradient')), rng=w.cpu(w.rng()),
            run=cfg['run_id'], config_sha256=w.sha(here/'config.json'),
            input_manifest_sha256=w.sha(here/'input_manifest.json'),
            applied_lrs=dict(first=cfg['first_lr'],later=cfg['later_lr']))

    def save(name):
        temp=out/(name+'.tmp');torch.save(snapshot(),temp);temp.replace(out/name)

    def guard():
        assert not stopped[0] and time.monotonic()-start < cfg['runtime_limit_seconds'] and not (here/'STOP').exists()

    def boundary(qualify=False):
        loss,g=c['task'].gradient();c['current_gradient']=g.clone();previous=c['previous_gradient']
        assert np.isfinite(loss) and bool(torch.isfinite(g).all())
        row=dict(state=c['state'],loss=loss,raw_gradient_norm=float(g.double().norm()),rho=cosine(g,previous),
            eta_kappa_hat=1-float(g.double()@previous.double())/float(previous.double()@previous.double()),
            active_gradient_norm=float(g[614600:].double().norm()),active_gradient_rho=cosine(g[614600:],previous[614600:]),
            first_gradient_norm=float(g[:614600].double().norm()),first_gradient_rho=cosine(g[:614600],previous[:614600]))
        if qualify:
            ref=refs[c['state']]
            assert row['loss']==ref['loss'] and row['raw_gradient_norm']==ref['raw_gradient_norm']
            for key in ['rho','eta_kappa_hat']:assert abs(row[key]-ref[key])<1e-12
        return row,g

    def capture(g):
        rng=w.rng();grads=[p.grad.clone() for p in model.parameters()]
        with torch.no_grad(): values=d.capture(model,x,cfg['physical_batch'],input_reference)
        assert values[-1]==5000;w.equal(rng,w.rng())
        for p,old in zip(model.parameters(),grads):assert torch.equal(p.grad,old)
        return dict(zip(all_names,[*values[:-1],g]))

    try:
        restore(parent);assert c['state']==448
        # Qualify the untreated trajectory and full optimizer/RNG state at512.
        checks=dict(status='running',exact_boundary_states=[],observer_preserves_rng_gradients=True)
        for state in range(448,513):
            guard();w.validate(c['opt'],state,.001)
            row,g=boundary(True);checks['exact_boundary_states'].append(state)
            w.append(out/'qualification_boundaries.jsonl',row)
            if state==448:
                current=capture(g)
                for n in all_names:assert np.array_equal(current[n].cpu().numpy(),prior[n][1]),n
            if state==512:
                saved=torch.load(historical/'prefix/end.pt',map_location='cpu',weights_only=False)
                current=snapshot()
                for key in ['model','optimizer','rng','previous_gradient','current_gradient']:w.equal(current[key],saved[key])
                save('qualification512.pt');checks['full_state512_exact']=True;break
            c['opt'].step();c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']==449:native_first=snapshot();torch.save(native_first,out/'native449.pt')
        restore(parent);save('fork448.pt')
        history={n:(torch.as_tensor(np.array(prior[n][0])).to('cuda:0'),None) for n in names}
        previous_update=None;last_loss=refs[447]['loss'];train_start=time.monotonic()
        checks.update(status='passed',exact_representations_at448=True)
        w.write(out/'qualification.json',checks)
        for state in range(448,802):
            guard();assert c['state']==state;w.validate(c['opt'],state,.001)
            row,g=boundary();row['loss_change']=row['loss']-last_loss;last_loss=row['loss']
            w.append(out/'boundaries.jsonl',row)
            current=capture(g)
            if state<=501:
                for n in all_names:arrays[n][state-447]=current[n].detach().cpu().numpy()
            rep={}
            for n in names:
                old,previous_delta=history[n];value=current[n];delta=value.double()-old.double()
                rep[n]=dict(change_rms=float(delta.square().mean().sqrt()),rho=cosine(delta,previous_delta) if previous_delta is not None else None,
                    activation_rms_from=float(old.double().square().mean().sqrt()),activation_rms_to=float(value.double().square().mean().sqrt()))
                history[n]=(value.clone(),delta)
            w.append(out/'representations.jsonl',dict(state=state-1,streams=rep))
            if state==801:break
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            layer_step(model,c['opt'],cfg['first_lr'],cfg['later_lr'])
            delta=torch.cat([(p.detach().double()-before[n].double()).flatten() for n,p in model.named_parameters()])
            layers={}
            for layer in [1,3,5]:
                dp=torch.cat([(p.detach().double()-before[f'{layer}.{n}'].double()).flatten() for n,p in model[layer].named_parameters()])
                states=[c['opt'].state[p] for p in model[layer].parameters()]
                moments=torch.cat([s['exp_avg'].double().flatten() for s in states])
                # Bias-corrected denominator used by this outgoing update.
                denominator=torch.cat([(s['exp_avg_sq'].double().sqrt()/np.sqrt(1-.999**(state+1))+1e-8).flatten() for s in states])
                layers[str(layer)]=dict(actual_increment_norm=float(dp.norm()),moment_norm=float(moments.norm()),
                    denominator_rms=float(denominator.square().mean().sqrt()),lr=cfg['first_lr'] if layer==1 else cfg['later_lr'])
            w.append(out/'updates.jsonl',dict(state=state,actual_increment_norm=float(delta.norm()),
                actual_increment_rho=cosine(delta,previous_update) if previous_update is not None else None,
                gradient_dot_actual_step=float(g.double()@delta),layers=layers))
            previous_update=delta;c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']==449:
                w.equal(w.cpu(c['opt'].state_dict()),native_first['optimizer']);w.equal(w.cpu(w.rng()),native_first['rng'])
                errors=[]
                for n,p in model.named_parameters():
                    factor=(cfg['first_lr'] if n.startswith('1.') else cfg['later_lr'])/.001
                    expected=parent['model'][n].double()+factor*(native_first['model'][n].double()-parent['model'][n].double())
                    error=float((p.detach().cpu().double()-expected).abs().max());errors.append(error)
                    torch.testing.assert_close(p.detach().cpu().double(),expected,rtol=0,atol=2e-7)
                    if factor==1:assert torch.equal(p.cpu(),native_first['model'][n])
                checks.update(first_step_all_moments_clocks_rng_exact=True,first_step_scaled_increment_max_abs_error=max(errors))
                w.write(out/'qualification.json',checks);save('intervention449.pt')
            if c['state']%64==0:
                save('recovery.pt')
                for a in arrays.values():a.flush()
                elapsed=time.monotonic()-train_start;speed=(c['state']-448)/elapsed
                event=dict(state=c['state'],loss=row['loss'],loss_state=state,elapsed_seconds=time.monotonic()-start,
                    updates_per_second=speed,estimated_remaining_seconds=(801-c['state'])/speed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save('end.pt')
        for a in arrays.values():a.flush()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        manifest.update(status='completed',completed_state=801,intervention_updates=353,dense_states=55,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),intervention_elapsed_seconds=time.monotonic()-train_start)
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))
