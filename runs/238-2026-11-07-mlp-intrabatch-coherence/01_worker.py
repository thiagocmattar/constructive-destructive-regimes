"""Exact Adam control replay with norm-based intra-batch gradient coherence."""
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
PARENT=ROOT/CFG['parent_run'];SOURCE=ROOT/CFG['parent_worker']
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
    manifest=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    w.write(out/'manifest.json',manifest)
    c=w.prepare('mlp',json.loads((PARENT/'config.json').read_text()))
    model=c['task'].model;x=c['task'].data['x_train'];y=c['task'].data['y_train']
    reference=w.rows(SOURCE/'prefix/boundaries.jsonl')[:-1]+w.rows(SOURCE/'control/boundaries.jsonl')
    energies=[];checked=[]
    try:
        for state in range(CFG['end']+1):
            assert not stopped[0] and time.monotonic()-start<CFG['runtime_limit_seconds'] and not (HERE/'STOP').exists()
            assert state==c['state']
            loss,g=c['task'].gradient();c['current_gradient']=g.clone();gn=float(g.double().norm())
            assert loss==reference[state]['loss'] and gn==reference[state]['raw_gradient_norm'],('replay mismatch',state)
            previous=c['previous_gradient'];rho=q=None
            if previous is not None:
                pg=previous.double();pn=float(pg.norm())
                rho=float(g.double()@pg)/(gn*pn) if gn*pn else None
                q=1-float(g.double()@pg)/pn**2 if pn else None
                assert abs(rho-reference[state]['rho'])<1e-12
                assert abs(q-reference[state]['eta_kappa_hat'])<1e-12
            row=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=rho,eta_kappa_hat=q,
                loss_change=loss-reference[state-1]['loss'] if state else None,exact_reference=True)
            w.append(out/'boundaries.jsonl',row)
            if state>=CFG['diagnostic_start']:
                rng=w.rng();grads=[p.grad.clone() for p in model.parameters()]
                e=d.example_energies(model,x,y,CFG['physical_batch']);record=d.coherence(gn,e)
                assert record['intrabatch_coherence'] is None or 0<=record['intrabatch_coherence']<=1+1e-6
                w.equal(rng,w.rng())
                for p,old in zip(model.parameters(),grads):w.equal(p.grad,old)
                w.append(out/'coherence.jsonl',dict(**row,**record));energies.append(e.cpu())
            if state in [512,800]:
                saved=torch.load(SOURCE/('prefix/end.pt' if state==512 else 'control/end.pt'),map_location='cpu',weights_only=False)
                for key,value in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(value),saved[key])
                checked.append(state);w.write(out/'qualification.json',dict(status='passed',exact_anchor_states=checked,observer_preserves_rng_gradients=True))
            if state==CFG['end']:break
            c['opt'].step();c['previous_gradient']=g.clone();c['state']=state+1;c['current_gradient']=None
            if c['state']%CFG['checkpoint_every']==0:save(c,out/'recovery.pt')
            if c['state']%64==0:
                elapsed=time.monotonic()-start
                event=dict(state=c['state'],loss_state=state,loss=loss,elapsed_seconds=elapsed,updates_per_second=c['state']/elapsed,
                    estimated_remaining_seconds=(CFG['end']-c['state'])*elapsed/c['state'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save(c,out/'end.pt')
        torch.save(dict(states=list(range(CFG['diagnostic_start'],CFG['end']+1)),energies=torch.stack(energies),examples=len(x)),out/'example_energies.pt')
        manifest.update(status='completed',updates=CFG['end'],diagnostic_states=len(energies),exact_anchor_states=checked,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
