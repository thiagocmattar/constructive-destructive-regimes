"""Exact Adam control replay with first preactivation, first-layer parameter histories and exact fixed inputs."""
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback
import torch
import numpy as np
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
    checked=[];input_checks=0;input_reference=x.detach().clone()
    prior_path=ROOT/'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/attempt-01/worker'
    prior=torch.load(prior_path/'outputs.pt',map_location='cpu',weights_only=False)
    previous240=ROOT/'runs/240-2026-11-07-mlp-hidden-change-svd/artifacts/attempt-01/worker'
    prior_h=np.load(previous240/'hidden.npy',mmap_mode='r');prior_g=np.load(previous240/'gradients.npy',mmap_mode='r')
    previous241=ROOT/'runs/241-2026-11-07-mlp-first-hidden-oscillation/artifacts/attempt-01/worker'
    prior_h1=np.load(previous241/'hidden1.npy',mmap_mode='r');prior_z2=np.load(previous241/'preactivation2.npy',mmap_mode='r')
    np.save(out/'inputs.npy',input_reference.flatten(1).cpu().numpy())
    count=CFG['end']-CFG['diagnostic_start']+1
    arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=np.float32,shape=shape)
            for key,shape in [('preactivation1',(count,5000,200)),('first_weights',(count,200,3072)),('first_biases',(count,200))]}
    w.write(out/'array_metadata.json',dict(states=list(range(CFG['diagnostic_start'],CFG['end']+1)),
        parameters=[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in model.named_parameters()],
        hidden='preactivation1: output of first linear before first tanh; inputs: flattened fixed pixels',dtype='float32',examples=5000,classes=10))
    try:
        for state in range(CFG['end']+1):
            assert not stopped[0] and time.monotonic()-start<CFG['runtime_limit_seconds'] and not (HERE/'STOP').exists()
            assert state==c['state']
            loss,g=c['task'].gradient();c['current_gradient']=g.clone();gn=float(g.double().norm())
            if state<len(reference):
                assert loss==reference[state]['loss'] and gn==reference[state]['raw_gradient_norm'],('replay mismatch',state)
            previous=c['previous_gradient'];rho=q=None
            if previous is not None:
                pg=previous.double();pn=float(pg.norm())
                rho=float(g.double()@pg)/(gn*pn) if gn*pn else None
                q=1-float(g.double()@pg)/pn**2 if pn else None
                if state<len(reference):
                    assert abs(rho-reference[state]['rho'])<1e-12
                    assert abs(q-reference[state]['eta_kappa_hat'])<1e-12
            row=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=rho,eta_kappa_hat=q,
                loss_change=loss-reference[state-1]['loss'] if state else None,exact_reference=state<len(reference))
            w.append(out/'boundaries.jsonl',row)
            if state>=CFG['diagnostic_start']:
                rng=w.rng();grads=[p.grad.clone() for p in model.parameters()]
                with torch.no_grad():
                    z1,hidden1,z2,hidden2,predictions,counted=d.capture(model,x,CFG['physical_batch'],input_reference)
                idx=state-CFG['diagnostic_start']
                assert torch.equal(predictions.cpu(),prior['outputs'][idx]),('output mismatch',state)
                assert np.array_equal(hidden2.cpu().numpy(),prior_h[idx]),('hidden2 mismatch',state)
                assert np.array_equal(g.cpu().numpy(),prior_g[idx]),('gradient mismatch',state)
                assert np.array_equal(hidden1.cpu().numpy(),prior_h1[idx]),('hidden1 mismatch',state)
                assert np.array_equal(z2.cpu().numpy(),prior_z2[idx]),('z2 mismatch',state)
                assert counted==len(x);input_checks+=1
                arrays['preactivation1'][idx]=z1.cpu().numpy()
                arrays['first_weights'][idx]=model[1].weight.detach().cpu().numpy()
                arrays['first_biases'][idx]=model[1].bias.detach().cpu().numpy()
                w.equal(rng,w.rng())
                for p,old in zip(model.parameters(),grads):w.equal(p.grad,old)

            if state in [512,800]:
                saved=torch.load(SOURCE/('prefix/end.pt' if state==512 else 'control/end.pt'),map_location='cpu',weights_only=False)
                for key,value in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(value),saved[key])
                checked.append(state);w.write(out/'qualification.json',dict(status='passed',exact_anchor_states=checked,observer_preserves_rng_gradients=True))
            if state==CFG['end']:
                saved=torch.load(prior_path/'end.pt',map_location='cpu',weights_only=False)
                for key,value in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(value),saved[key])
                checked.append(state)
                w.write(out/'qualification.json',dict(status='passed',exact_anchor_states=checked,all_355_outputs_exact=True,all_355_hidden2_and_gradients_exact=True,all_355_hidden1_and_z2_exact=True,all_355_inputs_exact=True,observer_preserves_rng_gradients=True))
                break
            c['opt'].step();c['previous_gradient']=g.clone();c['state']=state+1;c['current_gradient']=None
            if c['state']%CFG['checkpoint_every']==0:
                save(c,out/'recovery.pt')
                for array in arrays.values():array.flush()
            if c['state']%64==0:
                elapsed=time.monotonic()-start
                event=dict(state=c['state'],loss_state=state,loss=loss,elapsed_seconds=elapsed,updates_per_second=c['state']/elapsed,
                    estimated_remaining_seconds=(CFG['end']-c['state'])*elapsed/c['state'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        save(c,out/'end.pt')
        for array in arrays.values():array.flush()
        w.write(out/'input_verification.json',dict(status='passed',states_checked=input_checks,state_range=[447,801],samples_per_state=len(x),flattened_shape=list(input_reference.flatten(1).shape),raw_input_shape=list(x.shape),labels_sha256=w.tensor_hash([y]),input_sha256=w.tensor_hash([input_reference.flatten(1)]),max_input_change=0.0,check='Exact equality of every first-linear input batch against the initial fixed input in sample order; all355 states.'))
        manifest.update(status='completed',updates=CFG['end'],diagnostic_states=count,exact_anchor_states=checked,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        manifest.update(status='failed',exception=repr(error),traceback=traceback.format_exc());raise
    finally:
        manifest.update(completed_state=c['state'],finished_utc=w.stamp(),elapsed_seconds=time.monotonic()-start)
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
