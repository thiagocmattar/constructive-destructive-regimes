"""Replay exact squared-loss output-space terms at retained MLP audit states."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c
RUN_IDS=[269,272,275,276,277,278,279,282,283,284]


def main():
    c.w.configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    task=c.w.prepare('mlp',json.loads((ROOT/'runs/223-2026-11-04-multispike-interventions/config.json').read_text()))['task']
    targets=np.eye(10)[task.data['y_train'].cpu().numpy()];checks=[];maximum_loss_error=0.
    for number in RUN_IDS:
        run=next((ROOT/'runs').glob(str(number)+'-*'));worker=run/'artifacts/scientific/worker'
        manifest=json.loads((worker/'manifest.json').read_text());assert manifest['status']=='completed'
        for arm in manifest['arms']:
            folder=worker/arm;boundaries={r['state']:r for r in c.w.rows(folder/'boundaries.jsonl')}
            for path in sorted(folder.glob('audit_*.pt')):
                audit=torch.load(path,map_location='cpu',weights_only=False);outputs=[]
                for vector in [audit['before_parameters'],audit['actual_parameters']]:
                    offset=0
                    with torch.no_grad():
                        for p in task.model.parameters():p.copy_(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
                    assert offset==vector.numel()
                    values,_,_=c.policy.capture(task,'mlp',5000,[]);outputs.append(values['output'].cpu().numpy().astype(np.float64))
                old,new=outputs;delta=new-old;residual=old-targets;N,D=old.shape;t=audit['state']
                linear=float(np.sum(residual*delta)/N);quadratic=float(np.sum(delta*delta)/(2*N))
                before=float(np.sum(residual*residual)/(2*N));after=float(np.sum((new-targets)**2)/(2*N))
                np.testing.assert_allclose(linear+quadratic,after-before,rtol=1e-10,atol=2e-13)
                np.testing.assert_allclose(quadratic,D/2*boundaries[t+1]['representations']['output']['change_rms']**2,rtol=2e-8,atol=2e-11)
                np.testing.assert_allclose([before,after],[boundaries[t]['loss'],boundaries[t+1]['loss']],rtol=2e-7,atol=2e-9)
                error=max(abs(before-boundaries[t]['loss']),abs(after-boundaries[t+1]['loss']));maximum_loss_error=max(maximum_loss_error,error)
                checks.append(dict(run=number,arm=arm,state=t,N=N,D=D,linear=linear,quadratic=quadratic,loss_before_fp64=before,loss_after_fp64=after,loss_reduction_rounding_error=error))
        print(json.dumps(dict(run=number,status='passed')),flush=True)
    with Path(__file__).open('rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    c.w.write(HERE/'artifacts/mlp_loss_terms_audit.json',dict(status='passed',source_sha256=source,maximum_loss_reduction_rounding_error=maximum_loss_error,checks=checks))


if __name__=='__main__':main()
