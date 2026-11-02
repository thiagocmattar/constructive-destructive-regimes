"""Mechanics-only complete-objective Transformer accumulation calibration."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.root))
    from src.cifar_eos.core import environment,file_hash,tensor_hash
    from src.eos_tasks.production import ProductionTask
    from src.eos_tasks.task import configure,model_for
    configure();cache=a.root/'cache/eos-multitask';shared=json.loads((cache/'inputs.json').read_text())
    initial=shared['initial']['transformer'][0]
    assert file_hash(cache/initial['path'])==initial['sha256']
    data={k:v.cuda() for k,v in torch.load(cache/'transformer_data.pt',weights_only=True).items() if k in ('x_train','y_train')}
    model=model_for('transformer',cache,shared['wikitext']['vocabulary_size']).cuda()
    model.load_state_dict(torch.load(cache/initial['path'],weights_only=True))
    assert tensor_hash(model.parameters())==initial['initial_tensor_sha256']
    result=dict(mechanics_only=True,environment=environment('cuda:0'),samples=[])
    baseline=None;baseline_loss=None
    for physical in (64,128,256,512):
        task=ProductionTask('transformer',model,data,physical)
        for rep in range(2):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            begin=time.perf_counter();loss,g=task.gradient();torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            vector=g.double()
            if baseline is None:baseline=vector.clone();baseline_loss=loss
            relative=float((vector-baseline).norm()/baseline.norm())
            rho=float(vector.dot(baseline)/(vector.norm()*baseline.norm()))
            assert relative<=1e-5 and rho>=1-1e-8 and abs(loss-baseline_loss)<=1e-5
            row=dict(physical=physical,rep=rep,seconds=elapsed,loss=loss,
                relative_gradient_l2_error=relative,cosine_to_physical64=rho,
                objective_units=task.n,supervised_tokens=task.units,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            result['samples'].append(row);print(json.dumps(row),flush=True)
            a.output.write_text(json.dumps(result,indent=2)+'\n')
        assert tensor_hash(model.parameters())==initial['initial_tensor_sha256']
    result['status']='passed';a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
