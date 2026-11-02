"""One resumable GD reference or SGD fork; no cloud control or implicit jobs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import Geometry, append_json, environment, file_hash, flat, memory, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


@torch.no_grad()
def step(model, gradient, lr):
    cursor = 0
    for parameter in model.parameters():
        size = parameter.numel()
        parameter.add_(gradient[cursor:cursor+size].view_as(parameter), alpha=-lr)
        cursor += size
    assert cursor == gradient.numel()


def sampling_rng(task, seed, state, repetition, base):
    # Excluding batch size pairs permutation prefixes across batch interventions.
    return np.random.default_rng(np.random.SeedSequence([base, {'resnet': 1, 'transformer': 2}[task], seed, state, repetition]))


def atomic_torch(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def truncate(path, completed):
    kept = [r for r in rows(path) if r['update'] <= completed]
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in kept), encoding='utf-8', newline='\n')


def summarize(history):
    valid = [r['cosine'] for r in history if r['cosine'] is not None]
    last = history[-1]
    return dict(updates=len(history), valid_cosines=len(valid),
        mean_rho=float(np.mean(valid)) if valid else None, R=last['B_over_A'],
        actual_R=last['actual']['B_over_A'], absolute_displacement=last['actual_displacement'],
        max_nominal_displacement_relative_error=max(r['nominal_displacement_relative_error'] for r in history),
        max_actual_prefix_error=max(r['actual_prefix_error'] for r in history),
        sampled_units=sum(r['sampled_units'] for r in history),
        sampled_loss_units=sum(r['sampled_loss_units'] for r in history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['resnet', 'transformer'], required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--mode', choices=['reference', 'fork'], required=True)
    parser.add_argument('--state', type=int, choices=[0, 2000], default=0)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=3600)
    parser.add_argument('--smoke-updates', type=int, choices=[3])
    parser.add_argument('--stop-after', type=int, help='Recovery test only; requires --smoke-updates.')
    args = parser.parse_args()
    cfg = json.loads((HERE/'config.json').read_text())
    if args.stop_after and not args.smoke_updates:
        parser.error('--stop-after is mechanics only')
    if args.mode == 'fork' and (args.batch not in cfg['batches'][args.task] or not args.reference):
        parser.error('Fork requires a configured batch and explicit reference checkpoint.')
    batch = cfg['batches'][args.task][-1] if args.mode == 'reference' else args.batch
    if not 0 <= args.rep < (1 if batch == cfg['batches'][args.task][-1] else cfg['stochastic_repetitions']):
        parser.error('Repetition is outside the registered design.')
    inputs = json.loads((HERE/'input_manifest.json').read_text())
    for item in inputs['files']:
        if args.task in item['tasks']:
            path = ROOT/item['path']
            if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
                raise RuntimeError('Changed input: '+item['path'])
    configure(); device = torch.device('cuda:0'); torch.manual_seed(args.seed)
    cache = ROOT/cfg['cache']; shared = json.loads((cache/'inputs.json').read_text())
    data = {k: v.to(device) for k, v in torch.load(cache/(args.task+'_data.pt'), weights_only=True).items()
            if k in ('x_train', 'y_train')}
    model = model_for(args.task, cache, shared['wikitext']['vocabulary_size']).to(device)
    initial = shared['initial'][args.task][args.seed]
    model.load_state_dict(torch.load(cache/initial['path'], weights_only=True))
    assert tensor_hash(model.parameters()) == initial['initial_tensor_sha256']
    source = dict(path=(cache/initial['path']).relative_to(ROOT).as_posix(), sha256=file_hash(cache/initial['path']))
    if args.mode == 'fork':
        saved = torch.load(args.reference, map_location='cpu', weights_only=False)
        assert saved['state'] == args.state and saved['task'] == args.task and saved['seed'] == args.seed
        assert saved['mechanics_only'] == bool(args.smoke_updates)
        assert saved['config_sha256'] == file_hash(HERE/'config.json')
        model.load_state_dict(saved['model'])
        assert tensor_hash(model.parameters()) == saved['tensor_sha256']
        source = dict(path=args.reference.as_posix(), sha256=file_hash(args.reference))
    task = ProductionTask(args.task, model, data, cfg['physical'][args.task])
    assert task.n == cfg['batches'][args.task][-1]
    horizon = args.smoke_updates or cfg['reference_updates' if args.mode=='reference' else 'continuation_updates']
    identity = dict(task=args.task, seed=args.seed, mode=args.mode, state=args.state, batch=batch,
        repetition=args.rep, updates=horizon, mechanics_only=bool(args.smoke_updates),
        config_sha256=file_hash(HERE/'config.json'), input_manifest_sha256=file_hash(HERE/'input_manifest.json'),
        source=source)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        assert json.loads((args.output/'summary.json').read_text())['identity'] == identity
        print('Already complete and identity matches.'); return
    origin = flat(model.parameters()).detach().double()
    raw, actual = Geometry(origin.numel(), device), Geometry(origin.numel(), device)
    rng = sampling_rng(args.task, args.seed, args.state, args.rep, cfg['sampling_seed'])
    complete = 0; baseline = None; elapsed_prior = 0.
    recovery = args.output/'recovery.pt'
    if recovery.exists():
        saved = torch.load(recovery, map_location='cpu', weights_only=False)
        assert saved['identity'] == identity
        model.load_state_dict(saved['model']); raw.load_state_dict(saved['raw']); actual.load_state_dict(saved['actual'])
        origin = saved['origin'].to(device); rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        complete, baseline, elapsed_prior = saved['completed'], saved['baseline'], saved['elapsed_seconds']
        truncate(args.output/'metrics.jsonl', complete)
        truncate(args.output/'events.jsonl', complete)
        if batch != task.n:
            with (args.output/'indices.u32').open('r+b') as f: f.truncate(complete*batch*4)
    elif (args.output/'manifest.json').exists():
        raise RuntimeError('Incomplete attempt without recovery; preserve it and use a new output directory.')
    clock = time.perf_counter(); stopped = [False]
    for signum in (signal.SIGINT, signal.SIGTERM): signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    manifest = dict(identity=identity, status='running', started_utc=stamp(), environment=environment(device),
        objective_units=task.n, loss_units=task.units, starting_tensor_sha256=tensor_hash(model.parameters()),
        resumed_from_update=complete)
    write_json(args.output/'manifest.json', manifest)
    write_json(args.output/'resolved_config.json', cfg)
    lr = cfg['learning_rates'][args.task]

    def snapshot(path, state):
        atomic_torch(path, dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            task=args.task, seed=args.seed, state=state, mechanics_only=bool(args.smoke_updates),
            config_sha256=identity['config_sha256'], tensor_sha256=tensor_hash(model.parameters()),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), rng=rng.bit_generator.state))

    def save():
        atomic_torch(recovery, dict(identity=identity, model=model.state_dict(), raw=raw.state_dict(),
            actual=actual.state_dict(), origin=origin.cpu(), rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), completed=complete,
            baseline=baseline, elapsed_seconds=elapsed_prior+time.perf_counter()-clock))

    try:
        if baseline is None:
            baseline = task.evaluate()['loss']
            if args.mode == 'reference': snapshot(args.output/'state-0000.pt', 0)
            save()
        torch.cuda.reset_peak_memory_stats()
        with (args.output/'indices.u32').open('ab' if complete else 'wb') as index_file:
            for index in range(complete, horizon):
                if stopped[0] or (HERE/'STOP').exists() or time.perf_counter()-clock >= args.max_seconds:
                    break
                if args.stop_after and complete >= args.stop_after: break
                begin = time.perf_counter()
                ids = None
                if batch != task.n:
                    draw = rng.permutation(task.n)[:batch].astype('<u4')
                    index_file.write(draw.tobytes())
                    ids = torch.from_numpy(draw.astype(np.int64)).to(device)
                loss, g = task.gradient(ids)
                if not math.isfinite(loss): raise RuntimeError('Nonfinite sampled loss.')
                stats = raw.consume(g)
                before = flat(model.parameters()).detach().double()
                audit = args.seed == 0 and args.rep == 0 and index in (0, horizon-1)
                if audit: atomic_torch(args.output/f'audit-{index:04d}.pt', dict(gradient=g.cpu(), raw_prefix=raw.total.cpu(),
                    state=index, tensor_sha256=tensor_hash(model.parameters()), ids=None if ids is None else ids.cpu()))
                step(model, g, lr)
                after = flat(model.parameters()).detach().double()
                actual_stats = actual.consume(after-before)
                displacement = after-origin; norm = float(displacement.norm())
                error = float((displacement+lr*raw.total).norm())
                loss_units = task.units if ids is None else (batch*1000 if args.task=='resnet'
                    else int((data['y_train'][ids] != -100).sum()))
                complete = index+1
                row = dict(update=complete, sampled_loss=loss, **stats, actual=actual_stats,
                    sampled_units=batch, sampled_loss_units=loss_units, actual_displacement=norm,
                    nominal_displacement_error=error, nominal_displacement_relative_error=error/max(norm,1e-30),
                    actual_prefix_error=float((actual.total-displacement).norm()), seconds=time.perf_counter()-begin)
                append_json(args.output/'metrics.jsonl', row)
                event = dict(update=complete, requested_updates=horizon, utc=stamp(), loss=loss,
                    loss_scope='full_training' if batch==task.n else 'sampled_unbiased_objective',
                    rho=stats['cosine'], R=stats['B_over_A'], elapsed_seconds=elapsed_prior+time.perf_counter()-clock,
                    seconds_per_update=row['seconds'], **memory(device))
                append_json(args.output/'events.jsonl', event)
                if complete % cfg['checkpoint_every']==0 or complete==horizon:
                    index_file.flush(); save(); write_json(args.output/'progress.json', event)
                    print(json.dumps(event, allow_nan=False), flush=True)
        save()
        if complete == horizon:
            final_loss = task.evaluate()['loss']
            history = rows(args.output/'metrics.jsonl')
            assert [r['update'] for r in history] == list(range(1,horizon+1))
            summary = dict(identity=identity, **summarize(history), initial_loss=baseline, final_loss=final_loss,
                loss_change=final_loss-baseline, elapsed_seconds=elapsed_prior+time.perf_counter()-clock)
            assert summary['valid_cosines'] == horizon-1 and summary['R'] is not None
            assert math.isfinite(final_loss)
            snapshot(args.output/(f'state-{horizon:04d}.pt' if args.mode=='reference' else 'final.pt'),
                     horizon if args.mode=='reference' else args.state+horizon)
            summary['status'] = 'smoke_complete' if args.smoke_updates else 'complete'
            write_json(args.output/'summary.json', summary)
            # Completed forks need no rolling recovery; final tensors/RNG and raw evidence remain.
            recovery.unlink()
            manifest['status'] = summary['status']
        else:
            manifest['status'] = 'bounded_stop'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest.update(completed_updates=complete, finished_utc=stamp(), elapsed_seconds=elapsed_prior+time.perf_counter()-clock)
        write_json(args.output/'manifest.json', manifest)
        inventory = {p.name: dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in args.output.iterdir()
                     if p.is_file() and p.name != 'inventory.json'}
        write_json(args.output/'inventory.json', inventory)


if __name__ == '__main__':
    main()
