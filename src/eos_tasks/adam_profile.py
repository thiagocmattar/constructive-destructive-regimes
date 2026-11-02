"""Cold Adam forks of retained EOS states; exact anchored and adjacent gradients."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from src.cifar_eos.core import environment, file_hash, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for
from src.eos_tasks.lag_profile import cpu_products, gpu_products, atomic_array

ROOT = Path(__file__).resolve().parents[2]


def select_rate(results, rates):
    """Equal checkpoint weights; training-only selection, independent of cosine."""
    primary = [r for r in results if r['primary_eligible']]
    decisions = []
    for rate in rates:
        arms = [a for r in primary for a in r['arms'] if a['lr'] == rate]
        eligible = len(arms) == len(primary) and bool(arms)
        scores = []
        for a in arms:
            rows = a['rows']; base = rows[0]['loss']
            finite = a['status'] == 'complete' and all(math.isfinite(r['loss']) for r in rows)
            peak = max(r['loss'] for r in rows)
            terminal = float(np.mean([r['loss'] for r in rows[-16:]]))
            ok = finite and peak <= 4*base+1e-6 and terminal <= 1.05*base+1e-6
            eligible &= ok
            scores.append(math.log(max(terminal, 1e-30)/max(base, 1e-30)))
        decisions.append(dict(lr=rate, eligible=bool(eligible), checkpoints=len(arms),
            mean_log_terminal_loss_ratio=float(np.mean(scores)) if scores else None))
    valid = [r for r in decisions if r['eligible']]
    chosen = min(valid, key=lambda r:(r['mean_log_terminal_loss_ratio'], r['lr']))['lr'] if valid else None
    return dict(selected_lr=chosen, candidates=decisions, primary_checkpoints=len(primary))


def main(config_path):
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(config_path).read_text())
    configure(); torch.manual_seed(209)
    device = torch.device('cuda:0')
    stopping = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_:stopping.__setitem__(0, True))
    clock = time.perf_counter()
    manifest = dict(status='running', run_id=cfg['run_id'], worker=args.worker,
        started_utc=stamp(), config_sha256=file_hash(Path(config_path)), windows=[])

    def event(**fields):
        row = dict(utc=stamp(), elapsed_seconds=time.perf_counter()-clock,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), **fields)
        with (args.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        write_json(args.output/'progress.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    write_json(args.output/'manifest.json', manifest)
    try:
        for item in json.loads((ROOT/cfg['inputs_manifest']).read_text())['files']:
            if args.worker in item['workers']:
                path = ROOT/item['path']
                assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], item['path']
        write_json(args.output/'environment.json', environment(device))
        (args.output/'environment-lock.txt').write_text(os.popen(sys.executable+' -m pip freeze').read())
        shared = json.loads((ROOT/cfg['original_inputs']).read_text())
        for job in [j for j in cfg['jobs'] if j['worker'] == args.worker]:
            rid, name, anchor = job['parent_run_id'], job['task'], job['anchor']
            out = args.output/f'{rid}-{anchor}'; out.mkdir()
            data = {k:v.to(device) for k,v in torch.load(ROOT/job['data'], weights_only=True).items()
                if k.startswith(('x_train','y_train'))}
            model = model_for(name, ROOT/cfg['task_cache'], shared['wikitext']['vocabulary_size']).to(device)
            initial = torch.load(ROOT/job['checkpoint'], map_location='cpu', weights_only=False)
            assert initial['step'] == anchor
            reference = np.load(ROOT/job['anchor_gradient'], allow_pickle=False)
            model.load_state_dict(initial['model'])
            assert tensor_hash(model.parameters()) == job['tensor_sha256']
            task = ProductionTask(name, model, data, job['reference_physical'])
            loss, gradient = task.gradient()
            reference_gpu = torch.from_numpy(reference).to(device)
            error = float((gradient-reference_gpu).double().norm())/float(reference_gpu.double().norm())
            assert error < cfg['gradient_relative_tolerance'], ('source anchor gradient mismatch',error)
            checks = dict(source_anchor_relative_error=error, source_loss_error=loss-job['source_loss'])
            if job['physical'] != task.physical:
                baseline = gradient.clone(); baseline_loss = loss
                task.physical = job['physical']
                loss, gradient = task.gradient()
                error = float((gradient-baseline).double().norm())/float(baseline.double().norm())
                checks.update(compute_chunk_relative_error=error, compute_chunk_loss_error=loss-baseline_loss)
                assert error < cfg['gradient_relative_tolerance'] and abs(loss-baseline_loss) < 1e-5
                del baseline
            checks['fixed_tensor_sha256'] = tensor_hash(model.parameters())
            checks['resolved_physical'] = task.physical
            fixed_gradient = gradient.clone(); fixed_loss = loss
            anchor_cpu = gradient.detach().cpu().numpy().copy()
            anchor64 = gradient.double()
            atomic_array(out/'anchor.npy', anchor_cpu)
            write_json(out/'boundary_checks.json', checks)
            del gradient, reference_gpu
            summary = dict(parent_run_id=rid, task=name, parent_gd_lr=job['parent_gd_lr'], anchor=anchor,
                primary_eligible=job['primary_eligible'], initial_tensor_sha256=job['tensor_sha256'],
                source_checkpoint=job['checkpoint'], boundary_checks=checks, arms=[])
            for rate in job['rates']:
                arm = out/f'lr-{rate:g}'; arm.mkdir()
                model.load_state_dict(initial['model'])
                torch.set_rng_state(initial['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in initial['cuda_rng']])
                optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(.9,.999), eps=1e-8,
                    weight_decay=0, amsgrad=False, foreach=False, fused=False)
                assert len(optimizer.state) == 0
                # Restore p.grad explicitly: model.load_state_dict does not restore gradients.
                offset = 0
                for parameter in model.parameters():
                    count = parameter.numel()
                    parameter.grad = fixed_gradient[offset:offset+count].reshape_as(parameter).clone()
                    offset += count
                assert offset == fixed_gradient.numel()
                rows = []; previous_cpu = previous64 = None
                maximum = job['horizon']; status = 'complete'; reason = 'matched_gd_horizon'
                begin_arm = time.perf_counter()
                for lag in range(maximum+1):
                    begin = time.perf_counter()
                    if lag == 0:loss, gradient = fixed_loss, fixed_gradient
                    else:loss, gradient = task.gradient()
                    if not math.isfinite(loss) or not bool(torch.isfinite(gradient).all()):
                        status = 'nonfinite'; reason = 'nonfinite_loss_or_gradient'; break
                    current = gradient.detach().cpu().numpy().copy()
                    anchored = gpu_products(anchor64, gradient)
                    anchored_cpu = cpu_products(anchor_cpu, current)
                    adjacent = None if previous64 is None else gpu_products(previous64, gradient)
                    adjacent_cpu = None if previous_cpu is None else cpu_products(previous_cpu, current)
                    for gpu, cpu in [(anchored,anchored_cpu),(adjacent,adjacent_cpu)]:
                        if gpu is not None and gpu['cosine'] is not None:
                            assert abs(gpu['cosine']-cpu['cosine']) < cfg['cosine_audit_tolerance']
                            assert abs(gpu['norm']/cpu['norm']-1) < cfg['cosine_audit_tolerance']
                    row = dict(lag=lag, state=anchor+lag, loss=loss, norm=anchored['norm'],
                        anchor_cosine=anchored['cosine'], adjacent_cosine=None if adjacent is None else adjacent['cosine'],
                        anchor_cpu=anchored_cpu, adjacent_cpu=adjacent_cpu,
                        seconds=time.perf_counter()-begin)
                    rows.append(row)
                    if lag <= 2:atomic_array(arm/f'gradient-{lag}.npy', current)
                    halt = stopping[0] or (ROOT/cfg['stop_request']).exists()
                    if halt:status='bounded_stop';reason='budget_or_deadline'
                    if lag % cfg['recovery_every'] == 0 or lag == maximum or halt:
                        tmp = arm/'end.tmp'
                        torch.save(dict(step=anchor+lag, lag=lag, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), last_gradient_unapplied=True), tmp)
                        tmp.replace(arm/'end.pt')
                        atomic_array(arm/'gradient-current.npy',current)
                        if previous_cpu is not None:atomic_array(arm/'gradient-previous.npy',previous_cpu)
                        write_json(arm/'result.json',dict(lr=rate, status='running' if lag<maximum and not halt else status,
                            rows=rows, stop_reason=reason, elapsed_seconds=time.perf_counter()-begin_arm))
                    event(phase=cfg['phase'],parent=rid,task=name,anchor=anchor,lr=rate,lag=lag,
                        horizon=maximum, loss=loss, cosine=anchored['cosine'], adjacent_cosine=row['adjacent_cosine'],
                        seconds=time.perf_counter()-begin, completed_updates=lag,
                        loss_units=task.units, supervised_tokens=task.units if name=='transformer' else None)
                    if lag == maximum or halt:break
                    previous_cpu = current; previous64 = gradient.double()
                    optimizer.step()
                result = dict(lr=rate, status=status, rows=rows, stop_reason=reason,
                    last_lag=rows[-1]['lag'], last_gradient_unapplied=True,
                    elapsed_seconds=time.perf_counter()-begin_arm)
                write_json(arm/'result.json',result)
                summary['arms'].append(result)
                write_json(out/'result.json',summary)
                del optimizer
                if status == 'bounded_stop':break
            manifest['windows'].append(dict(parent_run_id=rid,anchor=anchor,arms=len(summary['arms'])))
            write_json(args.output/'manifest.json',manifest)
            del model, task, data, initial, fixed_gradient, anchor64, gradient, previous64
            torch.cuda.empty_cache()
            if stopping[0] or (ROOT/cfg['stop_request']).exists():break
        manifest.update(status='complete',completed_utc=stamp(),elapsed_seconds=time.perf_counter()-clock)
        write_json(args.output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),completed_utc=stamp())
        write_json(args.output/'manifest.json',manifest)
        raise


import sys
