"""Calibrate fixed gate thresholds exclusively from untreated updates416..447."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.eos_tasks.selective_first_linear import neuron_rms


def main():
    spec=importlib.util.spec_from_file_location('native',ROOT/'runs/223-2026-11-04-multispike-interventions/01_worker.py')
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    cfg=json.loads((HERE/'config.json').read_text())
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'],str(p)
    w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    out=HERE/'artifacts/attempt-01/worker';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stopped=[False]
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,lambda *_:stopped.__setitem__(0,True))
    manifest=dict(status='running',started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    w.write(out/'manifest.json',manifest)
    c=w.prepare('mlp',json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))
    reference=w.rows(ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/worker/prefix/boundaries.jsonl')
    model=c['task'].model;x=c['task'].data['x_train'].double().flatten(1)
    abs_updates=[];scores=[]
    try:
        for t in range(449):
            assert not stopped[0] and time.monotonic()-start<600 and not (HERE/'STOP').exists()
            assert c['state']==t;w.validate(c['opt'],t,.001)
            loss,g=c['task'].gradient();c['current_gradient']=g.clone()
            assert loss==reference[t]['loss'] and float(g.double().norm())==reference[t]['raw_gradient_norm']
            if t==448:
                parent=torch.load(ROOT/'runs/244-2026-11-07-mlp-freeze-first-linear/artifacts/attempt-01/worker/fork.pt',map_location='cpu',weights_only=False)
                for k,v in [('model',model.state_dict()),('optimizer',c['opt'].state_dict()),('rng',w.rng()),('previous_gradient',c['previous_gradient']),('current_gradient',g)]:w.equal(w.cpu(v),parent[k])
                torch.save(parent,out/'qualified448.pt');break
            before={n:p.detach().clone() for n,p in model[1].named_parameters()} if t>=416 else None
            c['opt'].step()
            if t>=416:
                dw=model[1].weight.detach().double()-before['weight'].double();db=model[1].bias.detach().double()-before['bias'].double()
                absolute=torch.cat([dw.flatten(),db.flatten()]).abs()
                value=neuron_rms(x,dw,db)
                abs_updates.append(absolute.cpu().numpy());scores.append(value.cpu().numpy())
                if t in [416,447]:torch.save(dict(state=t,weight=dw.cpu(),bias=db.cpu(),scores=value.cpu()),out/f'calibration_audit_{t}.pt')
            c['previous_gradient']=g.clone();c['current_gradient']=None;c['state']+=1
            if c['state']%64==0:
                event=dict(state=c['state'],loss=loss,elapsed_seconds=time.monotonic()-start,utc=w.stamp())
                w.write(out/'progress.json',event);w.append(out/'events.jsonl',event);print(json.dumps(event),flush=True)
        a=np.stack(abs_updates);s=np.stack(scores);assert a.shape==(32,614600) and s.shape==(32,200)
        np.save(out/'absolute_updates.npy',a);np.save(out/'neuron_rms.npy',s)
        thresholds=dict(status='verified',window=[416,447],states=list(range(416,448)),quantile_method='linear',
            coordinate_q90=float(np.quantile(a,.9,method='linear')),
            neuron={f'q{int(q*100)}':float(np.quantile(s,q,method='linear')) for q in [.9,.75,.5]},
            source='actual native FP32 Adam increments; products, RMS and quantiles in FP64',
            coordinate_shape=list(a.shape),neuron_shape=list(s.shape),all_examples=5000,
            config_sha256=w.sha(HERE/'config.json'),input_manifest_sha256=w.sha(HERE/'input_manifest.json'))
        w.write(out/'thresholds.json',thresholds)
        manifest.update(status='completed',completed_state=448,exact_boundary_count=449,full_fork448_exact=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        manifest.update(status='failed',exception=repr(e),traceback=traceback.format_exc());raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-start,finished_utc=w.stamp())
        w.write(out/'manifest.json',manifest)
        w.write(out/'inventory.json',dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=w.sha(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
