"""Post-hoc read-only audit of the exactly zero residual-stream channels."""
import argparse,hashlib,json
from pathlib import Path
import torch
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run=next((a.source_root/'runs').glob('294-*'));initial_path=a.data_root/'cache/eos-multitask/resnet_initial_seed0.pt';initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    rows=[];sources={str(initial_path):sha(initial_path)}
    for arm in ['control','cap_q25','cap_q50']:
        folder=next((run/'artifacts/attempt-01').glob('*/'+arm))
        for state in [256,288,576]:
            path=folder/(f'audit-{state:04d}.pt' if state!=576 else 'end.pt');payload=torch.load(path,map_location='cpu',weights_only=False);s=payload['before'] if state!=576 else payload;sources[str(path)]=sha(path)
            # State dict order contains parameters only: BN running buffers were removed.
            offset=0;gradient={}
            for n,v in s['model'].items():
                gradient[n]=s['current_gradient'][offset:offset+v.numel()].reshape(v.shape);offset+=v.numel()
            assert offset==s['current_gradient'].numel()
            for group,width in [(2,32),(3,64)]:
                for block in range(5):
                    base=f'layer{group}.{block}.bn2';idx=slice(16,width)
                    row=dict(arm=arm,state=state,site=f'layer{group}.{block}.relu#2',inactive_indices=list(range(16,width)),
                        initial_gamma_zero=bool((initial[base+'.weight']==0).all()),initial_beta_zero=bool((initial[base+'.bias']==0).all()),
                        retained_gamma_zero=bool((s['model'][base+'.weight'][idx]==0).all()),retained_beta_zero=bool((s['model'][base+'.bias'][idx]==0).all()),
                        gamma_gradient_zero=bool((gradient[base+'.weight'][idx]==0).all()),beta_gradient_zero=bool((gradient[base+'.bias'][idx]==0).all()))
                    assert all(v for k,v in row.items() if k.endswith('_zero')),row
                    rows.append(row)
    code=a.data_root/'cache/eos-multitask/reference/resnet_cifar.py';sources[str(code)]=sha(code)
    result=dict(status='passed',sources=sources,rows=rows,interpretation='Zero-initialized final branch BN plus zero-padded widening shortcuts and ReLU at zero explain the inactive post-add channels. Within-branch ReLU channels remain separate and need not be inactive. Post-hoc architectural audit, not a treatment effect.')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','audited_block_states':len(rows)}))
if __name__=='__main__':main()
