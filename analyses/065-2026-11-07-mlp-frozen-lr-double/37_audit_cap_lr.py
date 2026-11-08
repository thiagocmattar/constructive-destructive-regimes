"""H049 retained-state replay on the original Spark; never alters training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c
from src.cifar_eos.core import flat


def load_parameters(model, vector):
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    assert offset==vector.numel()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--worker',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text());worker=args.worker
    manifest=json.loads((worker/'manifest.json').read_text())
    assert manifest['status']=='completed' and all(r['status']=='completed' for r in manifest['results'])
    assert not manifest['mechanics_only'];started=time.monotonic()
    c.w.configure();threadpool_limits(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    task=c.w.prepare('mlp',json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))['task']
    checks=[];max_error=0.
    for arm in cfg['arms']:
        folder=worker/arm['id'];boundaries={r['state']:r for r in c.w.rows(folder/'boundaries.jsonl')}
        updates={r['state']:r for r in c.w.rows(folder/'updates.jsonl')}
        evals={r['state']:r for r in c.w.rows(folder/'evaluations.jsonl')}
        for path in sorted(folder.glob('audit_*.pt')):
            a=torch.load(path,map_location='cpu',weights_only=False);t=a['state'];history=[]
            for parameters in [*a['previous_parameters'],a['before_parameters']]:
                load_parameters(task.model,parameters)
                values,inputs,probe_loss=c.policy.capture(task,'mlp',cfg['probe'],['1','3'])
                history.append({n:v.cpu().numpy() for n,v in values.items()})
            np.testing.assert_allclose(probe_loss,boundaries[t]['probe_loss'],rtol=0,atol=0)
            records={}
            for n in history[0]:
                v=history[2][n].astype(np.float64);delta=v-history[1][n].astype(np.float64)
                old=history[1][n].astype(np.float64)-history[0][n].astype(np.float64)
                den=float(np.sqrt(np.sum(delta*delta)*np.sum(old*old)))
                stats=dict(activation_rms=float(np.sqrt(np.mean(v*v))),change_rms=float(np.sqrt(np.mean(delta*delta))),rho=float(np.sum(delta*old))/den if den else None)
                target=boundaries[t]['representations'][n];assert target['coordinates']==v.size
                for k,value in stats.items():
                    if value is None:assert target[k] is None
                    else:
                        np.testing.assert_allclose(value,target[k],rtol=2e-8,atol=2e-10)
                        max_error=max(max_error,abs(value-target[k]))
                records[n]=dict(shape=list(v.shape),**stats)
            # Replay the native optimizer before independently checking applied deltas.
            optimizer=torch.optim.Adam(task.model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
            optimizer.load_state_dict(copy.deepcopy(a['optimizer_before']))
            assert all(group['lr']==.001*arm['lr_multiplier'] for group in optimizer.param_groups)
            offset=0
            for p in task.model.parameters():
                p.grad=a['raw_gradient'][offset:offset+p.numel()].reshape_as(p).to(p.device).clone();offset+=p.numel()
            optimizer.step()
            assert torch.equal(flat(task.model.parameters()).cpu(),a['proposed_parameters'])
            c.w.equal(c.w.cpu(optimizer.state_dict()),a['optimizer_after'])
            offsets={};offset=0
            for name,shape in a['parameter_names']:
                size=int(np.prod(shape));offsets[name]=(slice(offset,offset+size),shape);offset+=size
            before=a['before_parameters'].numpy().astype(np.float64)
            proposed=a['proposed_parameters'].numpy().astype(np.float64)-before
            actual=a['actual_parameters'].numpy().astype(np.float64)-before
            local={}
            for name,gate in updates[t]['gates'].items():
                ws,shape=offsets[name+'.weight'];bs,_=offsets[name+'.bias']
                x=inputs[name].cpu().numpy().astype(np.float64)
                for key,delta in [('proposed_scores',proposed),('actual_scores',actual)]:
                    response=x@delta[ws].reshape(shape).T+delta[bs]
                    scores=np.sqrt(np.mean(response*response,axis=0))
                    np.testing.assert_allclose(scores,gate[key],rtol=2e-8,atol=2e-10)
                local[name]=dict(units=shape[0],threshold=gate['threshold'],max_actual_score=float(scores.max()))
            # Complete held-out evaluation at the same retained states.
            load_parameters(task.model,a['before_parameters'])
            test=task.evaluate('test');c.w.equal(test,evals[t]['splits']['test'])
            checks.append(dict(arm=arm['id'],state=t,native_adam_bitwise=True,optimizer_moments_bitwise=True,
                test_replay_exact=True,representations=records,local_response=local))
            print(json.dumps(dict(arm=arm['id'],state=t,passed=True)),flush=True)
    assert len(checks)==54
    with Path(__file__).open('rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    c.w.write(worker/'cap_lr_replay.json',dict(status='passed',source_sha256=source,checks=checks,
        method='Original-hardware parameter-triplet/native-Adam replay; independent NumPy representation and local affine reductions',
        max_representation_absolute_error=max_error,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()
