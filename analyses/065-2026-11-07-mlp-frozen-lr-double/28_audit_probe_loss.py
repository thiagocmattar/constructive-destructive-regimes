"""Independent NumPy replay of before/accepted probe losses for backtracking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--worker',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads((args.run/'config.json').read_text());manifest=json.loads((args.worker/'manifest.json').read_text());assert manifest['status']=='completed'
    c.w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    task=c.w.prepare(cfg['case'],json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))['task']
    y=task.data['y_train'][:cfg['probe']].cpu().numpy()
    if cfg['case']=='transformer':y=y[y!=-100]
    checks=[]
    for name in manifest['arms']:
        arm=next(a for a in cfg['arms'] if a['id']==name)
        if arm['kind']!='loss_guard':continue
        for path in sorted((args.worker/name).glob('audit_*.pt')):
            audit=torch.load(path,map_location='cpu',weights_only=False);losses=[]
            for vector in [audit['before_parameters'],audit['actual_parameters']]:
                offset=0
                with torch.no_grad():
                    for p in task.model.parameters():p.copy_(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
                assert offset==vector.numel();values,_,_=c.policy.capture(task,cfg['case'],cfg['probe'],[]);f=values['output'].cpu().numpy().astype(np.float64)
                losses.append(float(.5*np.sum((f-np.eye(10)[y])**2)/len(y)) if cfg['case']=='mlp' else float(-np.mean(f[np.arange(len(y)),y])))
            detail=next(g['loss_guard'] for g in audit['gates'].values() if 'loss_guard' in g)
            np.testing.assert_allclose(losses,[detail['before_loss'],detail['trials'][-1]['loss']],rtol=2e-10,atol=2e-12)
            assert losses[1]<=losses[0]+2e-12
            checks.append(dict(arm=name,state=audit['state'],before_loss=losses[0],actual_loss=losses[1]))
    assert len(checks)==3*sum(next(a for a in cfg['arms'] if a['id']==n)['kind']=='loss_guard' for n in manifest['arms'])
    with Path(__file__).open('rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    c.w.write(args.worker/'probe_loss_audit.json',dict(status='passed',source_sha256=source,checks=checks));print(json.dumps(dict(status='passed',checks=len(checks))))


if __name__=='__main__':main()
