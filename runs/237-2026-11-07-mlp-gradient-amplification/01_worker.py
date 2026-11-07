"""Unmodified Adam replay with backward, adaptive-scale and local-curvature audit."""
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback
import torch
import diagnostics as d

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CFG=json.loads((HERE/'config.json').read_text())
PARENT=ROOT/CFG['parent_run'];SOURCE=ROOT/CFG['parent_worker'];INTERNAL=ROOT/CFG['internal_worker']
spec=importlib.util.spec_from_file_location('w223',PARENT/'01_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def save(c,path):
    payload=dict(run=CFG['run_id'],config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'),
        completed_updates=c['state'],model=w.cpu(c['task'].model.state_dict()),optimizer=w.cpu(c['opt'].state_dict()),
        previous_gradient=w.cpu(c['previous_gradient']),current_gradient=w.cpu(c['current_gradient']),rng=w.rng())
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-start>CFG['runtime_limit_seconds'] or (HERE/'STOP').exists():raise RuntimeError('Bounded stop')
    manifest=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    w.write(out/'manifest.json',manifest)
    c=w.prepare('mlp',json.loads((PARENT/'config.json').read_text()))
    model=c['task'].model;x=c['task'].data['x_train'];y=c['task'].data['y_train']
    reference=w.rows(SOURCE/'prefix/boundaries.jsonl')[:-1]+w.rows(SOURCE/'control/boundaries.jsonl')
    cache=torch.load(ROOT/'cache/cifar10-eos/data.pt',map_location='cpu',weights_only=True)
    audit=dict(modules=[type(m).__name__ for m in model.modules()],input_normalization_std=d.summary(cache['normalization_std']),
        input_example_rms=d.summary(cache['x_train'].double().flatten(1).square().mean(1).sqrt()),
        no_dynamic_normalization=all(type(m).__name__ in ['Sequential','Flatten','Linear','Tanh'] for m in model.modules()))
    assert audit['no_dynamic_normalization'];w.write(out/'architecture_audit.json',audit);del cache
    residuals=[];tangents=[];second_tangents=[];checked=[]
    try:
        for state in range(CFG['end']+1):
            check();assert state==c['state']
            loss,g=c['task'].gradient();c['current_gradient']=g.clone();gn=float(g.double().norm())
            assert loss==reference[state]['loss'] and gn==reference[state]['raw_gradient_norm'],('replay mismatch',state)
            w.append(out/'boundaries.jsonl',dict(state=state,loss=loss,raw_gradient_norm=gn,exact_reference=True))
            active=state>=CFG['diagnostic_start']
            if active:
                rng=w.rng();old_gradients=[p.grad.clone() for p in model.parameters()]
                record,arrays=d.observe(model,x,y,CFG['physical_batch']);record.update(state=state,native_loss=loss)
                w.equal(rng,w.rng())
                for p,old in zip(model.parameters(),old_gradients):w.equal(p.grad,old)
                w.append(out/'backward.jsonl',record);residuals.append(arrays['residual'].cpu())
                if state in CFG['anchors']:
                    torch.save(dict(state=state,dz1=arrays['dz1'].cpu(),dz2=arrays['dz2'].cpu(),residual=arrays['residual'].cpu()),out/f'backward{state}.pt')
                del arrays,old_gradients
            if state in CFG['anchors']:
                saved=torch.load(INTERNAL/f'snapshots/state{state}.pt',map_location='cpu',weights_only=False)
                for key,value in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(value),saved[key])
                checked.append(state);w.write(out/'qualification.json',dict(status='passed',exact_anchor_states=checked,observer_preserves_rng_gradients=True))
            if state==CFG['end']:break
            if active:
                before={k:p.detach().clone() for k,p in model.named_parameters()};old_vectors=d.adam_vectors(c['opt'])
            c['opt'].step()
            if active:
                after={k:p.detach().clone() for k,p in model.named_parameters()}
                adam,delta=d.adam_observe(model,c['opt'],before,old_vectors,CFG['lr'],CFG['eps']);adam['state']=state
                bound=(CFG['betas'][1]*(1-CFG['betas'][1]**state)/(1-CFG['betas'][1]**(state+1)))**.5
                assert adam['minimum_denominator_ratio']>=bound-1e-7
                adam['denominator_ratio_lower_bound']=bound;w.append(out/'adam.jsonl',adam)
                model.load_state_dict(before);first=[];second=[];rs=[]
                with torch.no_grad():
                    for xb,yb in zip(x.split(CFG['physical_batch']),y.split(CFG['physical_batch'])):
                        a=d.forward(model,xb);v,z=d.directional(model,a,delta)
                        first.append(v);second.append(z);rs.append(a['output']-torch.nn.functional.one_hot(yb,10).to(x.dtype))
                model.load_state_dict(after)
                first,second,residual=torch.cat(first),torch.cat(second),torch.cat(rs)
                tangent_linear=float((residual.double()*first.double()).sum()/len(x))
                ggn=.5*float(first.double().square().sum()/len(x))
                residual_curvature=.5*float((residual.double()*second.double()).sum()/len(x))
                assert abs(tangent_linear-adam['gradient_dot_step'])<1e-8+1e-5*abs(adam['gradient_dot_step'])
                actual_change=reference[state+1]['loss']-loss
                record=dict(state=state,linear=adam['gradient_dot_step'],tangent_linear=tangent_linear,
                    gauss_newton_quadratic=ggn,residual_curvature_quadratic=residual_curvature,
                    predicted_loss_change=adam['gradient_dot_step']+ggn+residual_curvature,
                    actual_loss_change=actual_change,
                    ggn_over_descent=ggn/(-adam['gradient_dot_step']) if adam['gradient_dot_step']<0 else None,
                    directional_parameter_gain=(2*ggn)**.5/adam['actual_step_norm'])
                w.append(out/'directional.jsonl',record);tangents.append(first.cpu());second_tangents.append(second.cpu())
            c['previous_gradient']=g.clone();c['state']=state+1;c['current_gradient']=None
            if c['state']%CFG['checkpoint_every']==0:save(c,out/'recovery.pt')
            if c['state']%32==0:
                elapsed=time.monotonic()-start
                event=dict(state=c['state'],loss_state=state,loss=loss,elapsed_seconds=elapsed,updates_per_second=c['state']/elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save(c,out/'end.pt')
        torch.save(dict(states=list(range(448,536)),residuals=torch.stack(residuals),directional_states=list(range(448,535)),
            tangents=torch.stack(tangents),second_tangents=torch.stack(second_tangents),labels=y.cpu()),out/'output_vectors.pt')
        manifest.update(status='completed',updates=535,diagnostic_states=88,exact_anchor_states=checked,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
