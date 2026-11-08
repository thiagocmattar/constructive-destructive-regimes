"""Independent same-hardware replay of the actual functional budget at saved states."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--worker',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text());manifest=json.loads((args.worker/'manifest.json').read_text())
    assert manifest['status']=='completed' and not manifest['mechanics_only']
    c.w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    task=c.w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))['task'];checks=[]
    for name in manifest['arms']:
        arm=next(a for a in cfg['arms'] if a['id']==name)
        if arm['kind']!='budget':continue
        folder=args.worker/name
        for path in sorted(folder.glob('audit_*.pt')):
            audit=torch.load(path,map_location='cpu',weights_only=False);values={}
            detail=next(g['functional_budget'] for g in audit['gates'].values() if 'functional_budget' in g)
            for label in ['before','proposed','actual']:
                vector=audit[label+'_parameters'];offset=0
                with torch.no_grad():
                    for p in task.model.parameters():
                        p.copy_(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
                assert offset==vector.numel()
                captured,_,_=c.policy.capture(task,cfg['case'],cfg['probe'],[])
                values[label]={n:v.cpu().numpy().astype(np.float64) for n,v in captured.items() if n in detail['budgets']}
            measured={}
            for n,tau in detail['budgets'].items():
                measured[n]={}
                for label,key in [('proposed','native_ratios'),('actual','actual_ratios')]:
                    ratio=float(np.sqrt(np.mean((values[label][n]-values['before'][n])**2)))/tau
                    np.testing.assert_allclose(ratio,detail[key][n],rtol=2e-8,atol=2e-10)
                    if label=='actual':assert ratio<=1+1e-6
                    measured[n][label]=ratio
            checks.append(dict(arm=name,state=audit['state'],ratios=measured))
    assert len(checks)==3*sum(next(a for a in cfg['arms'] if a['id']==n)['kind']=='budget' for n in manifest['arms'])
    with Path(__file__).open('rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(status='passed',source_sha256=source,method='Saved before/native/actual parameters; independent NumPy probe RMS after same-hardware replay',checks=checks)
    c.w.write(args.worker/'functional_budget_audit.json',result);print(json.dumps(dict(status='passed',checks=len(checks))))


if __name__=='__main__':main()
