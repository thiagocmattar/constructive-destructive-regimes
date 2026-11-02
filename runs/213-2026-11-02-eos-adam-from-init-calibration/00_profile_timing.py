"""Local mechanics timing for cost estimation; never scientific calibration evidence."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from diagnostics import Diagnostics
from src.cifar_eos.core import environment, file_hash, flat, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates-only', action='store_true')
    args = parser.parse_args()
    configure()
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info()[0] < 5_000_000_000:
        raise RuntimeError('Insufficient idle device headroom for a local timing check.')
    cfg = json.loads((HERE/'config.json').read_text())
    cache = ROOT/cfg['cache']
    shared = json.loads((cache/'inputs.json').read_text())
    initial = shared['initial']['transformer'][0]
    datafile = cache/'transformer_data.pt'
    assert file_hash(datafile) == shared['prepared_files'][datafile.name]['sha256']
    assert file_hash(cache/initial['path']) == initial['sha256']
    data = {k: v.to(device) for k, v in torch.load(datafile, weights_only=True).items()
        if k.startswith(('x_train', 'y_train'))}
    model = model_for('transformer', cache, shared['wikitext']['vocabulary_size']).to(device)
    model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
    assert tensor_hash(model.parameters()) == initial['initial_tensor_sha256']
    task = ProductionTask('transformer', model, data, 64)
    result = dict(utc=stamp(), environment=environment(device), mechanics_only=True,
        supervised_tokens=task.units, sequences=task.n,
        initial_tensor_sha256=initial['initial_tensor_sha256'], chunks=[])
    for chunk in ([] if args.updates_only else [64]):
        task.physical = chunk
        torch.cuda.reset_peak_memory_stats()
        samples = []
        try:
            for index in range(3):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                _, gradient = task.gradient()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                samples.append(seconds)
                print(json.dumps(dict(phase='full_gradient', chunk=chunk, sample=index,
                    warmup=index == 0, seconds=seconds)), flush=True)
                del gradient
            result['chunks'].append(dict(chunk=chunk, status='complete',
                warmup_seconds=samples[0], measured_seconds=samples[1:],
                peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        except torch.OutOfMemoryError:
            result['chunks'].append(dict(chunk=chunk, status='out_of_memory', measured_seconds=samples))
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        write_json(args.output, result)
    # Measure full updates with the actual four streams at the prepared chunk64.
    task.physical = 64
    model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
    options = dict(cfg['optimizer']); options.pop('name')
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003, **options)
    diagnostics = Diagnostics(initial['parameters'], device)
    samples = []
    for index in range(3):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        _, gradient = task.gradient()
        before = flat(model.parameters()).detach().double()
        optimizer.step()
        after = flat(model.parameters()).detach().double()
        diagnostics.consume(gradient, optimizer, index+1, before, after)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-begin
        samples.append(seconds)
        print(json.dumps(dict(phase='full_update_four_streams', sample=index,
            warmup=index == 0, seconds=seconds)), flush=True)
    result['four_stream_updates'] = dict(chunk=64, warmup_seconds=samples[0],
        measured_seconds=samples[1:], peak_allocated_bytes=torch.cuda.max_memory_allocated())
    result['limits'] = ['Local laptop timings do not measure Runpod speed.',
        'Two steady samples per configuration; no sustained-throughput guarantee.',
        'No validation, checkpoint, provisioning or transfer timing is included.',
        'No scientific LR-regime or convergence evidence; no change to chunk64 training config.',
        'A separate chunk512 attempt was stopped under VRAM pressure; it supplies no timing estimate.']
    write_json(args.output, result)


if __name__ == '__main__':
    main()
