"""Run213 one-arm full-objective screening; no provisioning or LR interventions."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from diagnostics import Diagnostics, projected_change
from src.cifar_eos.core import append_json, environment, file_hash, flat, memory, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['mlp', 'resnet', 'transformer'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-updates', type=int, choices=[2], help='Two-update mechanics check; never paper evidence.')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    if args.lr not in cfg['learning_rates']:
        parser.error('LR is outside the frozen screening grid.')
    if args.smoke_updates:
        cfg['updates'] = args.smoke_updates
    inputs = json.loads((HERE/cfg['input_manifest']).read_text())
    for item in inputs['files']:
        if args.task not in item['tasks']:
            continue
        file = ROOT/item['path']
        if file.stat().st_size != item['bytes'] or file_hash(file) != item['sha256']:
            raise RuntimeError('Input changed: '+item['path'])
    configure()
    torch.manual_seed(cfg['seed'])
    device = torch.device(args.device)
    cache = ROOT/cfg['cache']
    shared = json.loads((cache/'inputs.json').read_text())
    datafile = ROOT/cfg['mlp_cache']/'data.pt' if args.task == 'mlp' else cache/(args.task+'_data.pt')
    data = {k: v.to(device) for k, v in torch.load(datafile, weights_only=True).items() if k.startswith(('x_', 'y_'))}
    model = model_for(args.task, cache, shared['wikitext']['vocabulary_size']).to(device)
    initial = shared['initial'][args.task][cfg['seed']]
    model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
    if tensor_hash(model.parameters()) != initial['initial_tensor_sha256']:
        raise RuntimeError('Frozen initial tensors do not match.')
    task = ProductionTask(args.task, model, data, cfg['physical'][args.task])
    options = dict(cfg['optimizer']); options.pop('name')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, **options)
    diagnostics = Diagnostics(initial['parameters'], device)
    initial_parameters = flat(model.parameters()).detach().double()
    start = 0
    baseline_loss = None
    identity = dict(config_sha256=file_hash(HERE/'config.json'),
        input_manifest_sha256=file_hash(HERE/cfg['input_manifest']), task=args.task, lr=args.lr,
        effective_updates=cfg['updates'], mechanics_only=bool(args.smoke_updates),
        source_sha256={p.name: file_hash(p) for p in [HERE/'01_train.py', HERE/'diagnostics.py']})
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise RuntimeError('Resume scientific identity differs.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        diagnostics.load_state_dict(saved['diagnostics'])
        initial_parameters = saved['initial_parameters'].to(device)
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, baseline_loss = saved['completed_updates'], saved['baseline_loss']
    args.output.mkdir(parents=True, exist_ok=False)
    stopping = [False]
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    clock = time.perf_counter()
    manifest = dict(identity, run_id=cfg['run_id'], status='running', started_utc=stamp(),
        full_batch=True, start_update=start, requested_updates=cfg['updates'], seed=cfg['seed'],
        initial_tensor_sha256=initial['initial_tensor_sha256'], environment=environment(device),
        objective_units=task.n, loss_units=task.units, physical=task.physical,
        resume_from=None if args.resume is None else dict(bytes=args.resume.stat().st_size, sha256=file_hash(args.resume)))
    write_json(args.output/'manifest.json', manifest)
    write_json(args.output/'resolved_config.json', dict(cfg, task=args.task, lr=args.lr))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    completed = start

    def save():
        target = args.output/'recovery.tmp'
        torch.save(dict(identity=identity, completed_updates=completed, model=model.state_dict(),
            optimizer=optimizer.state_dict(), diagnostics=diagnostics.state_dict(), baseline_loss=baseline_loss,
            initial_parameters=initial_parameters.cpu(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []), target)
        target.replace(args.output/'recovery.pt')

    try:
        status = 'smoke_complete' if args.smoke_updates else 'complete'
        for index in range(start, cfg['updates']+1):
            begin = time.perf_counter()
            loss, gradient = task.gradient()
            if not math.isfinite(loss) or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError('Nonfinite full-objective loss or raw gradient at state '+str(index))
            if baseline_loss is None:
                baseline_loss = loss
            boundary = dict(state=index, loss=loss, gradient_norm=float(gradient.double().norm()),
                projected_gradient_change=projected_change(diagnostics.streams['raw'].previous, gradient))
            append_json(args.output/'boundaries.jsonl', boundary)
            if index in (0, 1, cfg['updates']):
                torch.save(dict(state=index, raw_gradient=gradient.cpu()), args.output/('audit-gradient-'+str(index)+'.pt'))
            if index % cfg['evaluate_every'] == 0 or index == cfg['updates']:
                # Every prepared split is complete; no random or truncated probes.
                evaluation = {split: task.evaluate(split) for split in ('valid', 'test') if 'x_'+split in data}
                append_json(args.output/'evaluation.jsonl', dict(state=index, splits=evaluation, utc=stamp()))
            if index == cfg['updates']:
                break
            elapsed = time.perf_counter()-clock
            if stopping[0] or (HERE/cfg['stop_file']).exists() or elapsed >= cfg['maximum_arm_seconds']:
                status = 'bounded_stop'; break
            if loss > cfg['loss_guard_factor']*baseline_loss+1e-6:
                status = 'loss_guard'; break
            before = flat(model.parameters()).detach().double()
            # task.evaluate does not change gradients, BN state, RNG, or Adam clocks.
            optimizer.step()
            after = flat(model.parameters()).detach().double()
            stats = diagnostics.consume(gradient, optimizer, index+1, before, after)
            if not bool(torch.isfinite(after).all()):
                raise RuntimeError('Nonfinite parameters after update '+str(index+1))
            completed = index+1
            stats.update(update=completed, gradient_state=index, loss_before=loss,
                actual_parameter_displacement_norm=float((after-initial_parameters).norm()),
                actual_displacement_identity_residual_norm=float((diagnostics.streams['actual'].total-(after-initial_parameters)).norm()),
                seconds=time.perf_counter()-begin)
            append_json(args.output/'metrics.jsonl', stats)
            if completed % cfg['checkpoint_every'] == 0:
                save()
            elapsed = time.perf_counter()-clock
            event = dict(utc=stamp(), completed_updates=completed, requested_updates=cfg['updates'],
                loss=loss, loss_state=index, cumulative_loss_units=completed*task.units,
                supervised_tokens=completed*task.units if args.task == 'transformer' else None,
                examples_seen=completed*task.units if args.task != 'transformer' else None,
                elapsed_seconds=elapsed, updates_per_second=(completed-start)/elapsed,
                remaining_seconds_at_observed_rate=(cfg['updates']-completed)*elapsed/(completed-start),
                **memory(device))
            append_json(args.output/'events.jsonl', event)
            write_json(args.output/'progress.json', event)
            print(json.dumps(event, allow_nan=False), flush=True)
        save()
        if status in ('complete', 'smoke_complete'):
            (args.output/'recovery.pt').replace(args.output/'final.pt')
        manifest.update(status=status, completed_updates=completed, finished_utc=stamp(),
            elapsed_seconds=time.perf_counter()-clock, final_tensor_sha256=tensor_hash(model.parameters()))
    except BaseException as error:
        manifest.update(status='failed', completed_updates=completed, error=repr(error), finished_utc=stamp())
        # The last completed periodic recovery remains intact on numerical failure.
        manifest['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
            for p in args.output.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(args.output/'manifest.json', manifest)
        raise
    manifest['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p))
        for p in args.output.iterdir() if p.is_file() and p.name != 'manifest.json'}
    write_json(args.output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
