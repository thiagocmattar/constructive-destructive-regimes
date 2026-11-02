"""Two full-objective mechanics updates and start-gradient qualification; no paper evidence."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, flat, tensor_hash, write_json, environment
from src.eos_tasks.adam_diagnostics import Diagnostics
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['mlp','resnet','transformer'])
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    cfg = json.loads((HERE/'config.json').read_text())
    assert args.lr in cfg['learning_rates_by_task'][args.task]
    inputs = json.loads((HERE/'input_manifest.json').read_text())
    for item in inputs['files']:
        if args.task in item['tasks']:
            path = ROOT/item['path']
            assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256']
    configure(); torch.manual_seed(cfg['seed']); device=torch.device('cuda:0')
    cache=ROOT/cfg['cache']; shared=json.loads((cache/'inputs.json').read_text())
    datafile=ROOT/cfg['mlp_cache']/'data.pt' if args.task=='mlp' else cache/(args.task+'_data.pt')
    data={k:v.to(device) for k,v in torch.load(datafile,weights_only=True).items() if k.startswith(('x_','y_'))}
    model=model_for(args.task,cache,shared['wikitext']['vocabulary_size']).to(device)
    initial=shared['initial'][args.task][cfg['seed']]
    model.load_state_dict(torch.load(cache/initial['path'],weights_only=True))
    assert tensor_hash(model.parameters())==initial['initial_tensor_sha256']
    task=ProductionTask(args.task,model,data,cfg['physical'][args.task])
    options=dict(cfg['optimizer']);options.pop('name')
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,**options)
    diagnostics=Diagnostics(initial['parameters'],device)
    reference=torch.load(ROOT/cfg['qualification_reference_dir']/(args.task+'.pt'),weights_only=True)
    begin=time.perf_counter();rows=[]
    for index in range(2):
        start=time.perf_counter();loss,gradient=task.gradient()
        assert torch.isfinite(gradient).all() and loss>0
        if index==0:
            error=float((gradient.cpu().double()-reference['raw_gradient'].double()).norm()/reference['raw_gradient'].double().norm())
            assert error<1e-5,(args.task,error)
            assert abs(loss-reference['initial_loss'])/reference['initial_loss']<1e-6
            evaluations={split:task.evaluate(split) for split in ['valid','test'] if 'x_'+split in data}
            assert torch.equal(gradient,flat([p.grad for p in model.parameters()]))
            expected={'mlp':{'test':10000},'resnet':{'test':10000},'transformer':{'valid':217630,'test':245550}}
            assert {s:r['loss_units'] for s,r in evaluations.items()}==expected[args.task]
        before=flat(model.parameters()).detach().double();optimizer.step()
        after=flat(model.parameters()).detach().double()
        stats=diagnostics.consume(gradient,optimizer,index+1,before,after)
        assert torch.isfinite(after).all()
        assert all(stream.count==index+1 for stream in diagnostics.streams.values())
        rows.append(dict(update=index+1,loss=loss,seconds=time.perf_counter()-start,streams=stats['streams']))
    result=dict(status='passed',mechanics_only=True,task=args.task,lr=args.lr,updates=2,
        config_sha256=file_hash(HERE/'config.json'),input_manifest_sha256=file_hash(HERE/'input_manifest.json'),
        initial_tensor_sha256=initial['initial_tensor_sha256'],initial_gradient_relative_error=error,
        full_split_evaluations=evaluations,rows=rows,elapsed_seconds=time.perf_counter()-begin,
        environment=environment(device),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitation='Mechanics and numerical start qualification only; excluded from scientific trajectories.')
    write_json(args.output,result)
    print(json.dumps(dict(task=args.task,status='passed',relative_gradient_error=error,seconds=result['elapsed_seconds'])))


if __name__=='__main__':
    main()
