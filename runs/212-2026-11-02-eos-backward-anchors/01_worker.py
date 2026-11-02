"""Exact retrospective anchored cosines from dense 256-update continuations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import environment, file_hash, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for
from src.eos_tasks.lag_profile import cpu_products, gpu_products


def backward_rows(history, start, device):
    """The last gradient is fixed; increasing lag indexes earlier vectors."""
    last = len(history)-1
    anchor = np.asarray(history[last])
    fixed = torch.from_numpy(anchor.copy()).to(device).double()
    rows = []
    for lag in range(1, last+1):
        earlier = np.asarray(history[last-lag])
        gpu = gpu_products(fixed, torch.from_numpy(earlier.copy()).to(device))
        cpu = cpu_products(anchor, earlier)
        if gpu['cosine'] is None:
            assert cpu['cosine'] is None
        else:
            assert abs(gpu['cosine']-cpu['cosine']) < 1e-10
        rows.append(dict(lag=lag, reference_state=start+last,
                         compared_state=start+last-lag, cosine=gpu['cosine'],
                         defined=gpu['cosine'] is not None, gpu=gpu, cpu=cpu))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfgpath = Path(__file__).parent/'config.json'
    cfg = json.loads(cfgpath.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    configure(); torch.manual_seed(209)
    device = torch.device('cuda:0')
    stopping = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    begin = time.perf_counter()
    manifest = dict(status='running', worker=args.worker, run_id='212',
                    started_utc=stamp(), config_sha256=file_hash(cfgpath), jobs=[])
    write_json(args.output/'manifest.json', manifest)

    def event(**values):
        row = dict(utc=stamp(), elapsed_seconds=time.perf_counter()-begin,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), **values)
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(args.output/'progress.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        for item in json.loads((ROOT/cfg['inputs_manifest']).read_text())['files']:
            if args.worker in item['workers']:
                path = ROOT/item['path']
                assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256']
        write_json(args.output/'environment.json', environment(device))
        (args.output/'environment-lock.txt').write_text(os.popen(sys.executable+' -m pip freeze').read())
        shared = json.loads((ROOT/cfg['original_inputs']).read_text())
        for job in [j for j in cfg['jobs'] if j['worker'] == args.worker]:
            out = args.output/job['id']; out.mkdir()
            initial = torch.load(ROOT/job['checkpoint'], map_location='cpu', weights_only=False)
            assert initial['step'] == job['start']
            data = {k:v.to(device) for k,v in torch.load(ROOT/job['data'], weights_only=True).items()
                    if k.startswith(('x_train','y_train'))}
            model = model_for(job['task'], ROOT/cfg['task_cache'], shared['wikitext']['vocabulary_size']).to(device)
            model.load_state_dict(initial['model'])
            assert tensor_hash(model.parameters()) == job['tensor_sha256']
            torch.set_rng_state(initial['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in initial['cuda_rng']])
            task = ProductionTask(job['task'], model, data, job['reference_physical'])
            loss, reference = task.gradient()
            saved = torch.from_numpy(np.load(ROOT/job['anchor_gradient'])).to(device)
            source_error = float((reference-saved).double().norm()/saved.double().norm())
            checks = dict(source_gradient_relative_error=source_error,
                          source_loss_error=loss-job['source_loss'], initial_tensor_sha256=tensor_hash(model.parameters()))
            write_json(out/'boundary_checks.json', checks)
            assert source_error < 1e-5 and abs(loss-job['source_loss']) < 1e-5, checks
            task.physical = job['physical']
            actual_loss, gradient = task.gradient()
            chunk_error = float((gradient-reference).double().norm()/reference.double().norm())
            checks.update(compute_chunk_relative_error=chunk_error, compute_chunk_loss_error=actual_loss-loss,
                          resolved_physical=task.physical)
            write_json(out/'boundary_checks.json', checks)
            assert chunk_error < 1e-5 and abs(actual_loss-loss) < 1e-5, checks
            del saved, reference
            if job['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=job['lr'], betas=(.9,.999),
                    eps=1e-8, weight_decay=0, amsgrad=False, foreach=False, fused=False)
                assert not optimizer.state
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=job['lr'], momentum=0,
                                            weight_decay=0, foreach=False)
                optimizer.load_state_dict(initial['optimizer'])
                assert all(g['lr'] == job['lr'] and not g['momentum'] for g in optimizer.param_groups)
            del initial
            count = gradient.numel(); horizon = cfg['horizon']
            history = np.lib.format.open_memmap(out/'gradients.npy', mode='w+', dtype=np.float32,
                                                shape=(horizon+1, count))
            scalars = []; first64 = gradient.double(); previous = None
            for offset in range(horizon+1):
                tick = time.perf_counter()
                if offset:
                    actual_loss, gradient = task.gradient()
                assert math.isfinite(actual_loss) and bool(torch.isfinite(gradient).all())
                current_cpu = gradient.detach().cpu().numpy()
                history[offset] = current_cpu
                raw = gradient.double()
                scalar = dict(offset=offset, state=job['start']+offset, loss=actual_loss,
                    norm=float(raw.norm()), gradient_sha256=hashlib.sha256(current_cpu.tobytes()).hexdigest(),
                    forward_cosine=gpu_products(first64, gradient)['cosine'],
                    adjacent_cosine=None if previous is None else gpu_products(previous, gradient)['cosine'])
                scalars.append(scalar)
                halt = stopping[0] or (ROOT/cfg['stop_request']).exists()
                if offset % 32 == 0 or offset == horizon or halt:
                    history.flush()
                    torch.save(dict(step=job['start']+offset, offset=offset, model=model.state_dict(),
                        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), last_gradient_unapplied=True), out/'end.pt')
                    write_json(out/'scalars.json', dict(rows=scalars))
                event(phase='gradient', job=job['id'], task=job['task'], optimizer=job['optimizer'],
                      start=job['start'], endpoint=job['start']+horizon, offset=offset, horizon=horizon,
                      loss=actual_loss, norm=scalar['norm'], seconds=time.perf_counter()-tick,
                      loss_units=task.units, supervised_tokens=task.units if job['task']=='transformer' else None)
                if halt: raise RuntimeError('bounded stop request; preserve partial history and recovery')
                if offset == horizon: break
                previous = raw
                optimizer.step()
            history.flush()
            event(phase='backward_products', job=job['id'])
            rows = backward_rows(history, job['start'], device)
            retained = [0, 128, 192, 254, 255, 256]
            np.save(out/'audit_gradients.npy', np.asarray(history[retained]))
            write_json(out/'result.json', dict(job=job, endpoint=job['start']+horizon,
                gradients=horizon+1, applied_updates=horizon, boundary_checks=checks,
                parameter_count=count, retained_offsets=retained, dense_history_retained=False,
                scalar_rows=scalars, rows=rows, status='complete'))
            manifest['jobs'].append(job['id']); write_json(args.output/'manifest.json', manifest)
            del history, model, optimizer, task, data, gradient, first64, previous, raw
            (out/'gradients.npy').unlink()
            torch.cuda.empty_cache()
        manifest.update(status='complete', completed_utc=stamp(), elapsed_seconds=time.perf_counter()-begin)
        write_json(args.output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), completed_utc=stamp())
        write_json(args.output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
