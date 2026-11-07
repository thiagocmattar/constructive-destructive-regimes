"""Exact Adam replay with passive internal observations and static probes."""
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback
import torch
import diagnostics as d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CFG=json.loads((HERE/'config.json').read_text())
PARENT=ROOT/CFG['parent_run'];SOURCE=ROOT/CFG['parent_worker']
spec=importlib.util.spec_from_file_location('frozen223',PARENT/'01_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def save(c,path):
    p=dict(run=CFG['run_id'],config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'),
        completed_updates=c['state'],model=w.cpu(c['task'].model.state_dict()),optimizer=w.cpu(c['opt'].state_dict()),
        rng=w.rng(),previous_gradient=w.cpu(c['previous_gradient']),current_gradient=w.cpu(c['current_gradient']))
    temp=path.with_suffix('.tmp');torch.save(p,temp);temp.replace(path)


def main():
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker';out.mkdir(parents=True,exist_ok=False)
    snapshots=out/'snapshots';snapshots.mkdir()
    start=time.monotonic();stopped=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0] or time.monotonic()-start>CFG['runtime_limit_seconds'] or (HERE/'STOP').exists():raise RuntimeError('Bounded stop')
    status=dict(status='running',phase='replay',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    w.write(out/'manifest.json',status)
    reference=w.rows(SOURCE/'prefix/boundaries.jsonl')[:-1]+w.rows(SOURCE/'control/boundaries.jsonl')
    anchors=sorted({512,535}|{v for e in CFG['events'] for v in [e['before'],e['peak']-1,e['peak'],e['recovery']]})
    c=w.prepare('mlp',json.loads((PARENT/'config.json').read_text()))
    model=c['task'].model;x=c['task'].data['x_train'];y=c['task'].data['y_train']
    try:
        for state in range(CFG['end']+1):
            check();assert state==c['state']
            loss,g=c['task'].gradient();gn=float(g.double().norm());c['current_gradient']=g.clone()
            assert loss==reference[state]['loss'] and gn==reference[state]['raw_gradient_norm'],('replay mismatch',state,loss,reference[state]['loss'])
            w.append(out/'boundaries.jsonl',dict(state=state,loss=loss,raw_gradient_norm=gn,exact_reference=True))
            if state>=CFG['diagnostic_start']:
                rng=w.rng();gradients=[p.grad.clone() for p in model.parameters()]
                record,arrays=d.observe(model,c['opt'],x,y,CFG['physical_batch'],state in anchors)
                record.update(state=state,loss=loss);w.append(out/'internals.jsonl',record)
                w.equal(rng,w.rng())
                for p,old_g in zip(model.parameters(),gradients):w.equal(p.grad,old_g)
                assert abs(record['output']['loss_fp64']-loss)<2e-7
                if state in anchors:
                    save(c,snapshots/f'state{state}.pt')
                    torch.save(dict(state=state,arrays=w.cpu(arrays),labels=y.cpu()),snapshots/f'activations{state}.pt')
                del arrays,gradients
            if state==512:
                own=torch.load(snapshots/'state512.pt',map_location='cpu',weights_only=False)
                original=torch.load(SOURCE/'prefix/end.pt',map_location='cpu',weights_only=False)
                for key in ['model','optimizer','rng','previous_gradient','current_gradient']:w.equal(own[key],original[key])
                w.write(out/'qualification.json',dict(status='passed',all_prefix_losses_norms_exact=True,exact_state512_model_optimizer_rng_gradients=True,observer_gradient_rng_unchanged=True))
            if state==CFG['end']:break
            before={k:p.detach().clone() for k,p in model.named_parameters()} if state>=CFG['diagnostic_start'] else None
            c['opt'].step()
            if before is not None:
                steps={}
                for key,p in model.named_parameters():
                    delta=p.detach().double()-before[key].double();raw=p.grad.double();m=c['opt'].state[p]['exp_avg'].double()
                    steps[key]=dict(norm=float(delta.norm()),relative_norm=float(delta.norm()/before[key].double().norm()),
                        gradient_dot_step=float((raw*delta).sum()),gradient_step_cosine=d.cosine(raw,delta),
                        momentum_norm=float(m.norm()),gradient_momentum_cosine=d.cosine(raw,m))
                w.append(out/'steps.jsonl',dict(state=state,groups=steps,gradient_dot_step=sum(q['gradient_dot_step'] for q in steps.values())))
            c['previous_gradient']=g.clone();c['state']=state+1;c['current_gradient']=None
            if c['state']%CFG['checkpoint_every']==0:save(c,out/'recovery.pt')
            if c['state']%32==0:
                elapsed=time.monotonic()-start
                event=dict(phase='replay',state=c['state'],loss_state=state,loss=loss,elapsed_seconds=elapsed,
                    updates_per_second=c['state']/elapsed,remaining_replay_seconds=(CFG['end']-c['state'])*elapsed/c['state'],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save(c,out/'end.pt')
        status.update(phase='static_probes',replay_updates=CFG['end'],replay_elapsed_seconds=time.monotonic()-start,anchors=anchors)
        w.write(out/'manifest.json',status)
        def get(state):
            p=torch.load(snapshots/f'state{state}.pt',map_location='cuda:0',weights_only=False)
            a=torch.load(snapshots/f'activations{state}.pt',map_location='cuda:0',weights_only=False)
            return p,a['arrays']
        for event in CFG['events']:
            for phase,from_state,to_state in [('rise',event['before'],event['peak']),('recovery',event['peak'],event['recovery'])]:
                check();a,aa=get(from_state);b,bb=get(to_state)
                record=dict(event=event['name'],phase=phase,from_state=from_state,to_state=to_state,
                    hybrid=d.hybrids(model,a['model'],b['model'],x,y,CFG['physical_batch']),
                    output=d.output_decomposition(a['model'],b['model'],aa,bb,y),
                    changes=d.changes(a['model'],b['model'],aa,bb))
                assert record['hybrid']['loss_by_mask'][0]==reference[from_state]['loss']
                assert record['hybrid']['loss_by_mask'][-1]==reference[to_state]['loss']
                w.append(out/'counterfactuals.jsonl',record)
                event_record=dict(phase='static_probes',event=event['name'],comparison=phase,elapsed_seconds=time.monotonic()-start,utc=w.stamp())
                w.write(out/'progress.json',event_record);print(json.dumps(event_record),flush=True)
            check();a,_=get(event['peak']-1);b,_=get(event['peak'])
            delta=w.flat(b['model'].values()).double()-w.flat(a['model'].values()).double()
            linear=float(a['current_gradient'].double().dot(delta));curve=[]
            for alpha in [i/16 for i in range(25)]:
                params=a['model'] if alpha==0 else b['model'] if alpha==1 else {k:a['model'][k]+alpha*(b['model'][k]-a['model'][k]) for k in d.GROUPS}
                model.load_state_dict(params);curve.append(dict(scale=alpha,loss=d.native_loss(model,x,y,CFG['physical_batch'])))
            assert curve[0]['loss']==reference[event['peak']-1]['loss'] and curve[16]['loss']==reference[event['peak']]['loss']
            actual=curve[16]['loss']-curve[0]['loss']
            w.append(out/'step_curves.jsonl',dict(event=event['name'],from_state=event['peak']-1,to_state=event['peak'],
                linear_prediction=linear,actual_loss_change=actual,nonlinear_remainder=actual-linear,curve=curve))
        status.update(status='completed',phase='done',peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        status.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',status)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
