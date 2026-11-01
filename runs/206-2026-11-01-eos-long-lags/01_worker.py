"""Streaming exact anchored cosines; full-objective continuations of Run205."""
import argparse
import json
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


def cpu_products(anchor, gradient):
    """Independent NumPy FP64 reduction, bounded temporary memory."""
    aa = bb = ab = 0.0
    for start in range(0, len(anchor), 65536):
        a = np.asarray(anchor[start:start+65536], dtype=np.float64)
        b = np.asarray(gradient[start:start+65536], dtype=np.float64)
        aa += float(np.dot(a, a)); bb += float(np.dot(b, b)); ab += float(np.dot(a, b))
    return dict(dot=ab, anchor_norm=aa**.5, norm=bb**.5,
                cosine=ab/(aa*bb)**.5 if aa > 0 and bb > 0 else None)


def gpu_products(anchor64, gradient):
    b = gradient.double()
    aa, bb, ab = float(anchor64.dot(anchor64)), float(b.dot(b)), float(anchor64.dot(b))
    return dict(dot=ab, anchor_norm=aa**.5, norm=bb**.5,
                cosine=ab/(aa*bb)**.5 if aa > 0 and bb > 0 else None)


def envelope(rows, width=64):
    """Median half-gap of adjacent odd/even branches; descriptive stopping rule."""
    by_lag = {r['lag']: r['cosine'] for r in rows}
    last = max(by_lag)
    values = []
    for even in range(last-width+2, last+1, 2):
        a, b = by_lag.get(even-1), by_lag.get(even)
        if a is not None and b is not None:
            values.append(abs(b-a)/2)
    return float(np.median(values)) if len(values) == width//2 else None


def atomic_array(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--physical', type=int, default=64)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((Path(__file__).parent/'config.json').read_text())
    configure(); torch.manual_seed(206)
    device = torch.device('cuda:0')
    clock = time.perf_counter()
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    manifest = dict(status='running', worker=args.worker, started_utc=stamp(),
                    config_sha256=file_hash(Path(__file__).parent/'config.json'), windows=[])
    write_json(args.output/'manifest.json', manifest)

    def event(**values):
        r = dict(utc=stamp(), elapsed_seconds=time.perf_counter()-clock,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), **values)
        with (args.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(r, allow_nan=False)+'\n')
        write_json(args.output/'progress.json', r)
        print(json.dumps(r, allow_nan=False), flush=True)

    try:
        for item in json.loads((ROOT/cfg['inputs_manifest']).read_text())['files']:
            if args.worker in item['workers']:
                path = ROOT/item['path']
                assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], item['path']
        write_json(args.output/'environment.json', environment(device))
        (args.output/'environment-lock.txt').write_text(os.popen(sys.executable+' -m pip freeze').read())
        shared = json.loads((ROOT/cfg['original_inputs']).read_text())
        jobs = [j for j in cfg['jobs'] if j['worker'] == args.worker]
        if args.calibrate:
            jobs = [j for j in jobs if j['task'] == 'transformer']
            assert len(jobs) == 1
        for job in jobs:
            rid, name, anchor = job['parent_run_id'], job['task'], job['anchor']
            out = args.output/f'{rid}-{anchor}'
            out.mkdir()
            data = {k:v.to(device) for k,v in torch.load(ROOT/job['data'], weights_only=True).items()
                    if k.startswith(('x_train','y_train'))}
            model = model_for(name, ROOT/cfg['task_cache'], shared['wikitext']['vocabulary_size']).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=job['lr'], momentum=0, weight_decay=0, foreach=False)
            checkpoint = torch.load(ROOT/job['checkpoint'], map_location='cpu', weights_only=False)
            assert checkpoint['step'] == anchor+10
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            torch.set_rng_state(checkpoint['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
            assert tensor_hash(model.parameters()) == job['resume_tensor_sha256']
            del checkpoint
            anchor_cpu = np.load(ROOT/job['anchor_gradient'], allow_pickle=False)
            boundary_cpu = np.load(ROOT/job['boundary_gradient'], allow_pickle=False)
            anchor64 = torch.from_numpy(anchor_cpu).to(device).double()
            task = ProductionTask(name, model, data, job['physical'])
            # Recompute the unapplied lag-10 gradient at the exact retained end state.
            loss, gradient = task.gradient()
            boundary = torch.from_numpy(boundary_cpu).to(device)
            boundary_error = float((gradient-boundary).double().norm())/float(boundary.double().norm())
            assert boundary_error < cfg['gradient_relative_tolerance'], boundary_error
            checks = dict(source_boundary_relative_error=boundary_error, source_boundary_loss_error=loss-job['prefix_rows'][-1]['loss'])
            if args.calibrate:
                reference = gradient.clone(); reference_loss = loss
                candidates = []
                for physical in cfg['transformer_compute_candidates']:
                    task.physical = physical
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); times = []
                    errors = []
                    try:
                        for repeat in range(2):
                            begin = time.perf_counter(); trial_loss, trial = task.gradient()
                            torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
                            error = float((trial-reference).double().norm())/float(reference.double().norm())
                            errors.append(error)
                        peak = torch.cuda.max_memory_allocated()
                        passed = max(errors) < cfg['gradient_relative_tolerance'] and abs(trial_loss-reference_loss) < 1e-5
                        passed = passed and peak < .8*torch.cuda.get_device_properties(device).total_memory
                        candidates.append(dict(physical=physical, passed=passed, seconds=times,
                            relative_gradient_errors=errors, loss_error=trial_loss-reference_loss, peak_allocated_bytes=peak))
                    except torch.cuda.OutOfMemoryError:
                        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                        candidates.append(dict(physical=physical, passed=False, warning='out_of_memory'))
                    event(phase='calibration', parent=rid, anchor=anchor, candidate=candidates[-1])
                valid = [c for c in candidates if c['passed']]
                assert valid
                selected = min(valid, key=lambda c: np.median(c['seconds']))['physical']
                write_json(args.output/'calibration.json', dict(selected_physical=selected, candidates=candidates,
                    mathematical_objective_unchanged=True, fp32_accumulation_partition_changes=True, checks=checks))
                manifest.update(status='complete', completed_utc=stamp())
                break
            if name == 'transformer' and args.physical != job['physical']:
                reference = gradient
                task.physical = args.physical
                loss, gradient = task.gradient()
                checks['compute_chunk_relative_error'] = float((gradient-reference).double().norm())/float(reference.double().norm())
                assert checks['compute_chunk_relative_error'] < cfg['gradient_relative_tolerance']
                del reference
            checks['resolved_physical'] = task.physical
            write_json(out/'boundary_checks.json', checks)
            rows = [dict(r) for r in job['prefix_rows']]
            previous_cpu = boundary_cpu
            stages = []
            stop_reason = None
            previous_stage = 10
            fixed_start = time.perf_counter()
            for lag in range(10, cfg['stages'][-1]+1):
                begin = time.perf_counter()
                if lag > 10:
                    loss, gradient = task.gradient()
                assert bool(torch.isfinite(gradient).all()), 'Nonfinite full gradient'
                current_cpu = gradient.detach().cpu().numpy()
                gpu = gpu_products(anchor64, gradient)
                cpu = cpu_products(anchor_cpu, current_cpu)
                if gpu['cosine'] is not None:
                    assert abs(gpu['cosine']-cpu['cosine']) < cfg['cosine_audit_tolerance']
                    assert abs(gpu['norm']/cpu['norm']-1) < cfg['cosine_audit_tolerance']
                if lag > 10:
                    row = dict(lag=lag, state=anchor+lag, loss=loss, **gpu, cpu=cpu,
                               seconds=time.perf_counter()-begin)
                    rows.append(row)
                    if lag == 11:
                        atomic_array(out/'audit-first.npy', current_cpu)
                if stop[0] or (ROOT/cfg['stop_request']).exists():
                    stop_reason = 'budget_or_deadline_request'
                if lag in cfg['stages']:
                    gap = envelope(rows, cfg['envelope_width'])
                    stages.append(dict(lag=lag, median_half_odd_even_gap=gap))
                    # Positive-lag-1 controls have no initial reversal to damp.
                    # All profiles have the same minimum 256-lag coverage.
                    if not job['initial_reversal']:
                        stop_reason = stop_reason or 'positive_lag1_control_minimum_horizon'
                    elif gap is not None and gap <= cfg['damped_gap_threshold']:
                        stop_reason = stop_reason or 'alternating_envelope_below_threshold'
                    elif lag == cfg['stages'][-1]:
                        stop_reason = stop_reason or 'maximum_horizon_reached'
                    previous_stage = lag
                event(phase='gradient', parent=rid, task=name, anchor=anchor, lag=lag,
                      completed_updates=anchor+lag, loss=loss, cosine=gpu['cosine'],
                      norm=gpu['norm'], seconds=time.perf_counter()-begin,
                      objective_units=task.n, loss_units=task.units,
                      supervised_tokens=task.units if name == 'transformer' else None,
                      resolved_physical=task.physical)
                if lag % cfg['recovery_every'] == 0 or stop_reason:
                    tmp = out/'end.tmp'
                    torch.save(dict(step=anchor+lag, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), tmp)
                    tmp.replace(out/'end.pt')
                    atomic_array(out/'audit-previous.npy', previous_cpu)
                    atomic_array(out/'audit-current.npy', current_cpu)
                    result = dict(parent_run_id=rid, task=name, lr=job['lr'], anchor=anchor,
                        last_lag=lag, last_state=anchor+lag, primary_eligible=job['primary_eligible'],
                        source_job=job['source_window'], source_resume_sha256=job['checkpoint_sha256'],
                        boundary_checks=checks, rows=rows, stages=stages, stop_reason=stop_reason,
                        initial_reversal=job['initial_reversal'],
                        sampled_raw_lags=[11,lag-1,lag], last_gradient_unapplied=True,
                        elapsed_seconds=time.perf_counter()-fixed_start)
                    write_json(out/'result.json', result)
                if stop_reason:
                    break
                previous_cpu = current_cpu.copy()
                optimizer.step()
            manifest['windows'].append(dict(parent_run_id=rid, anchor=anchor, last_lag=lag, stop_reason=stop_reason))
            write_json(args.output/'manifest.json', manifest)
            del task, model, optimizer, data, anchor64, gradient, boundary
            torch.cuda.empty_cache()
            if stop_reason == 'budget_or_deadline_request':
                break
        manifest.update(status='complete', completed_utc=stamp(), elapsed_seconds=time.perf_counter()-clock)
        write_json(args.output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), completed_utc=stamp())
        write_json(args.output/'manifest.json', manifest)
        raise
    finally:
        write_json(args.output/'inventory.json', dict(files=[dict(path=p.relative_to(args.output).as_posix(),
            bytes=p.stat().st_size, sha256=file_hash(p)) for p in sorted(args.output.rglob('*'))
            if p.is_file() and p.name != 'inventory.json']))


if __name__ == '__main__':
    main()
