"""Replay retained parameter triplets; audit whole-probe reductions with NumPy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--worker',type=Path,required=True);args=a.parse_args()
    run=args.run;worker=args.worker;cfg=json.loads((run/'config.json').read_text())
    manifest=json.loads((worker/'manifest.json').read_text());assert manifest['status']=='completed' and not manifest['mechanics_only']
    c.w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    task=c.w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))['task']
    checked=[];max_abs=0.
    for name in manifest['arms']:
        folder=worker/name;boundaries={r['state']:r for r in c.w.rows(folder/'boundaries.jsonl')}
        for path in sorted(folder.glob('audit_*.pt')):
            audit=torch.load(path,map_location='cpu',weights_only=False);history=[]
            for parameters in [*audit['previous_parameters'],audit['before_parameters']]:
                offset=0
                with torch.no_grad():
                    for p in task.model.parameters():
                        p.copy_(parameters[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
                assert offset==parameters.numel()
                values,_,_=c.policy.capture(task,cfg['case'],cfg['probe'],[])
                history.append({n:v.cpu().numpy() for n,v in values.items()});del values
            t=audit['state'];records={}
            for n in history[0]:
                v=history[2][n].astype(np.float64);delta=v-history[1][n].astype(np.float64);old=history[1][n].astype(np.float64)-history[0][n].astype(np.float64)
                den=float(np.sqrt(np.sum(delta*delta)*np.sum(old*old)))
                stats=dict(activation_rms=float(np.sqrt(np.mean(v*v))),change_rms=float(np.sqrt(np.mean(delta*delta))),rho=float(np.sum(delta*old))/den if den else None)
                target=boundaries[t]['representations'][n];assert target['coordinates']==v.size
                for k,value in stats.items():
                    if value is None:assert target[k] is None
                    else:
                        np.testing.assert_allclose(value,target[k],rtol=2e-8,atol=2e-10)
                        max_abs=max(max_abs,abs(value-target[k]))
                records[n]=dict(shape=list(v.shape),**stats)
            checked.append(dict(arm=name,state=t,representations=records))
            print(json.dumps(dict(arm=name,state=t,passed=True)),flush=True)
    with Path(__file__).open('rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    c.w.write(worker/'representation_audit.json',dict(status='passed',source_sha256=source,method='Same-hardware replay of saved parameter triplets; independent NumPy whole-probe reductions',max_absolute_error=max_abs,checks=checked))


if __name__=='__main__':main()
