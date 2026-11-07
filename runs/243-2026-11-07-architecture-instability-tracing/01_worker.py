"""Unchanged Adam continuations with full objectives and fixed activation probes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import diagnostics as d

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT))
CFG = json.loads((HERE/'config.json').read_text()); PARENT = ROOT/CFG['parent']
spec = importlib.util.spec_from_file_location('w223', PARENT/'01_worker.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


def context(case):
    c = w.prepare(case, json.loads((PARENT/'config.json').read_text()))
    p = HERE/'inputs'/case/'prefix.pt'
    w.load(c, p)
    # CUDA generator state formats can differ between devices; checkpoint load
    # is recorded separately from measured historical replay agreement.
    return c


def capture(c, folder, state, arrays, metadata, first=False):
    case = c['case']; sp = CFG['cases'][case]; task = c['task']; model = task.model
    x = task.data['x_train'][:sp['probe']]; y = task.data['y_train'][:sp['probe']]
    mask = (y != -100) if case == 'transformer' else None
    rng = w.rng(); grads = [p.grad.clone() for p in model.parameters()]
    buffers = {n: v.clone() for n, v in model.named_buffers()}
    i = state - (sp['start']-1); count = sp['end']-sp['start']+3
    def emit(name, a):
        if name not in arrays:
            number = len(arrays); path = folder/'raw'/f'{number:03d}.npy'
            arrays[name] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, *a.shape))
            metadata[name] = dict(path=path.name, shape=list(a.shape), coordinates=int(a.size))
        arrays[name][i] = a
    with torch.no_grad():
        with d.Capture(model, case, emit, mask) as observer:
            value, output = task.loss(x, y)
        emit('output.log_probabilities', (output if case=='transformer' else torch.log_softmax(output, dim=-1)).detach().cpu().numpy()[mask.cpu().numpy()] if case=='transformer' else torch.log_softmax(output, dim=-1).cpu().numpy())
        if first:
            plain_loss, plain = task.loss(x, y)
            assert torch.equal(plain, output) and torch.equal(plain_loss, value)
            np.save(folder/'raw'/'fixed_inputs.npy', x.cpu().numpy())
            np.save(folder/'raw'/'fixed_targets.npy', y.cpu().numpy())
        else:
            assert np.array_equal(x.cpu().numpy(), np.load(folder/'raw'/'fixed_inputs.npy'))
    w.equal(rng, w.rng())
    for p, g in zip(model.parameters(), grads): assert torch.equal(p.grad, g)
    for n, v in model.named_buffers(): assert torch.equal(v, buffers[n])
    torch.save(w.cpu(model.state_dict()), folder/'checkpoints'/f'{state}.pt')
    emit('raw_gradient', c['current_gradient'].cpu().numpy())
    denom = int(mask.sum()) if mask is not None else len(y)
    return float(value.double()/denom)


def execute(case, calibration=False):
    w.configure(); torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    for item in json.loads((HERE/'input_manifest.json').read_text())['files']:
        p=ROOT/item['path']; assert p.stat().st_size==item['bytes'] and w.sha(p)==item['sha256'], str(p)
    out = HERE/'artifacts'/('calibration' if calibration else 'attempt-01')/case
    out.mkdir(parents=True, exist_ok=False); (out/'raw').mkdir(); (out/'checkpoints').mkdir()
    sp=CFG['cases'][case]; start=time.monotonic(); c=context(case)
    status=dict(status='running',case=case,started_utc=w.stamp(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                coverage=dict(training_units=c['task'].units,objective_units=c['task'].n,probe_units=sp['probe']))
    w.write(out/'manifest.json',status); arrays={}; metadata={}; comparisons=[]
    reference={r['state']:r for r in w.rows(HERE/'inputs'/case/'boundaries.jsonl')}
    end=c['state']+2 if calibration else sp['end']+1
    try:
        while c['state'] <= end:
            assert time.monotonic()-start < CFG['seconds_per_case'] and not (HERE/'STOP').exists()
            state=c['state']; tick=time.monotonic(); loss,g=c['task'].gradient(); c['current_gradient']=g.clone()
            assert np.isfinite(loss) and bool(torch.isfinite(g).all())
            gn=float(g.double().norm()); pg=c['previous_gradient']; pn=float(pg.double().norm()) if pg is not None else 0
            rho=float(g.double()@pg.double())/(gn*pn) if gn*pn else None
            if state in reference:
                r=reference[state]; comparisons.append(dict(state=state,loss_exact=loss==r['loss'],norm_exact=gn==r['raw_gradient_norm'],
                    loss_relative_error=abs(loss-r['loss'])/max(abs(r['loss']),1e-30)))
            probe=None
            if calibration or state >= sp['start']-1:
                # Calibration captures one production-shaped state at the first
                # allocated slot; it never enters the scientific evidence.
                probe=capture(c,out,sp['start']-1 if calibration else state,arrays,metadata,first=(not metadata))
            row=dict(state=state,loss=loss,raw_gradient_norm=gn,rho=rho,probe_loss=probe,elapsed_seconds=time.monotonic()-start,step_seconds=time.monotonic()-tick)
            w.append(out/'boundaries.jsonl',row); w.write(out/'progress.json',row); print(json.dumps(row),flush=True)
            if state==end: break
            c['opt'].step(); c['previous_gradient']=g.clone(); c['state']+=1
            if state%16==0: w.save(c,out/'recovery.pt',dict(id='unchanged'))
        w.save(c,out/'end.pt',dict(id='unchanged'))
        for a in arrays.values(): a.flush()
        w.write(out/'stages.json',metadata)
        w.write(out/'historical_comparison.json',dict(exact=all(r['loss_exact'] and r['norm_exact'] for r in comparisons),rows=comparisons,
            interpretation='Same saved fork, unchanged policy; Spark continuation, not assumed bitwise historical trajectory'))
        if not calibration:
            grams={}; levels={}
            with threadpool_limits(limits=4):
                for k,(name,meta) in enumerate(metadata.items()):
                    a=np.load(out/'raw'/meta['path'],mmap_mode='r')
                    # Gradients need states a-1..b; the extra b+1 is only for representation changes.
                    path=out/'raw'/meta['path']
                    if name=='raw_gradient':
                        temp=out/'raw'/'gradient_fit.npy'; np.save(temp,a[:-1]); path=temp
                    gram,level=d.gram_reduce(path,raw_gradient=name=='raw_gradient')
                    grams[name]=gram; levels[name]=level
                    w.write(out/'reduction_progress.json',dict(stage=name,completed=k+1,total=len(metadata)))
            np.savez(out/'grams.npz',**grams); np.savez(out/'levels.npz',**levels)
        status.update(status='completed',completed_state=c['state'],capture_states=[sp['start']-1,sp['end']+1],observer_checks='RNG, buffers, gradients unchanged; plain forward parity; fixed input exact',
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),raw_bytes=sum(p.stat().st_size for p in (out/'raw').glob('*')))
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-start,finished_utc=w.stamp()); w.write(out/'manifest.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--case',choices=['resnet','transformer'],required=True); p.add_argument('--calibrate',action='store_true'); a=p.parse_args()
    execute(a.case,a.calibrate)
