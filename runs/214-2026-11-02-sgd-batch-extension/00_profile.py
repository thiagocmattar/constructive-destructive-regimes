"""Bounded local production-shaped mechanics/timing, never paper evidence."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import environment, file_hash, flat, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['resnet', 'transformer'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    args = p.parse_args()
    assert not args.output.exists(), 'Preserve calibration attempts.'
    cfg = json.loads((HERE/'config.json').read_text())
    cache = ROOT/cfg['cache']
    shared = json.loads((cache/'inputs.json').read_text())
    datafile = cache/(args.task+'_data.pt')
    assert file_hash(datafile) == shared['prepared_files'][datafile.name]['sha256']
    initial = shared['initial'][args.task][0]
    assert file_hash(cache/initial['path']) == initial['sha256']
    configure()
    device = torch.device('cuda:0')
    data = {k: v.to(device) for k, v in torch.load(datafile, weights_only=True).items()
            if k in ('x_train', 'y_train')}
    model = model_for(args.task, cache, shared['wikitext']['vocabulary_size']).to(device)
    model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
    assert tensor_hash(model.parameters()) == initial['initial_tensor_sha256']
    task = ProductionTask(args.task, model, data, cfg['physical'][args.task])
    sizes = cfg['batches'][args.task] if args.full else cfg['batches'][args.task][:-1]
    result = dict(mechanics_only=True, task=args.task, utc=stamp(), environment=environment(device),
                  config_sha256=file_hash(HERE/'config.json'), samples=[], objective_units=task.n,
                  loss_units=task.units, source_sha256=file_hash(__file__))
    total = torch.zeros(sum(p.numel() for p in model.parameters()), device=device, dtype=torch.float64)
    previous = None
    try:
        for b in sizes:
            for repetition in range(3):
                model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
                ids = None if b == task.n else torch.arange(b, device=device)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                begin = time.perf_counter()
                loss, g = task.gradient(ids)
                before = flat(model.parameters()).detach().double()
                g64 = g.double(); total += g64
                norm = float(g64.norm()); retention = float(total.norm())
                cosine = None if previous is None else float(previous.dot(g64)/(previous.norm()*g64.norm()))
                with torch.no_grad():
                    cursor = 0
                    for parameter in model.parameters():
                        parameter.add_(g[cursor:cursor+parameter.numel()].view_as(parameter), alpha=-cfg['learning_rates'][args.task])
                        cursor += parameter.numel()
                after = flat(model.parameters()).detach().double()
                update_norm = float((after-before).norm())
                previous = g64
                torch.cuda.synchronize()
                sample = dict(batch=b, repetition=repetition, warmup=repetition==0,
                    seconds=time.perf_counter()-begin, loss=loss, gradient_norm=norm,
                    actual_update_norm=update_norm, diagnostic_sum_norm=retention, cosine=cosine,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                result['samples'].append(sample); print(json.dumps(sample), flush=True)
                write_json(args.output, result)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_utc'] = stamp(); write_json(args.output, result)


if __name__ == '__main__':
    main()
