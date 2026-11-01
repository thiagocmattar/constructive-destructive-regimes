"""Exact, anchored lag profiles from frozen full-GD checkpoints (Run205)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import environment, file_hash, flat, stamp, tensor_hash, write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure, model_for


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b)) / denominator if denominator > 0 else None


def close(actual, expected, absolute, relative):
    return abs(actual - expected) <= absolute + relative * abs(expected)


def qualify(row, reference, tolerances):
    checks = {
        'loss': close(row['loss'], reference['loss'], tolerances['loss_absolute'], tolerances['loss_relative']),
        'norm': close(row['norm'], reference['norm'], tolerances['norm_absolute'], tolerances['norm_relative']),
    }
    if row.get('adjacent_cosine') is not None and reference.get('cosine') is not None:
        checks['adjacent_cosine'] = close(row['adjacent_cosine'], reference['cosine'], tolerances['cosine_absolute'], 0)
    return dict(state=row['state'], passed=all(checks.values()), checks=checks,
                loss_error=row['loss']-reference['loss'], norm_error=row['norm']-reference['norm'],
                cosine_error=None if 'adjacent_cosine' not in checks else row['adjacent_cosine']-reference['cosine'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((Path(__file__).parent/'config.json').read_text())
    manifest_path = args.output/'manifest.json'
    manifest = dict(status='running', worker=args.worker, started_utc=stamp(), config_sha256=file_hash(Path(__file__).parent/'config.json'))
    write_json(manifest_path, manifest)
    clock = time.perf_counter()

    def event(**values):
        record = dict(utc=stamp(), elapsed_seconds=time.perf_counter()-clock, **values)
        if torch.cuda.is_available():
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        write_json(args.output/'progress.json', record)
        print(json.dumps(record, allow_nan=False), flush=True)

    try:
        configure()
        torch.manual_seed(205)
        device = torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats()
        inputs = json.loads((ROOT/cfg['inputs_manifest']).read_text())
        shared = json.loads((ROOT/cfg['original_inputs']).read_text())
        for item in inputs['files']:
            path = ROOT/item['path']
            assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], item['path']
        write_json(args.output/'environment.json', environment(device))
        (args.output/'environment-lock.txt').write_text(os.popen(sys.executable+' -m pip freeze').read())
        completed = []
        for condition in cfg['conditions']:
            if condition['worker'] != args.worker:
                continue
            rid, task_name = condition['parent_run_id'], condition['task']
            out = args.output/rid
            out.mkdir()
            data = {k:v.to(device) for k,v in torch.load(ROOT/condition['data'], weights_only=True).items() if k.startswith(('x_train','y_train'))}
            model = model_for(task_name, ROOT/cfg['task_cache'], shared['wikitext']['vocabulary_size']).to(device)
            task = ProductionTask(task_name, model, data, condition['physical'])
            optimizer = torch.optim.SGD(model.parameters(), lr=condition['lr'], momentum=0, weight_decay=0, foreach=False)
            references = json.loads((ROOT/condition['reference']).read_text())
            reference_rows = {r['state']:r for r in references['rows']}
            condition_result = dict(parent_run_id=rid, task=task_name, lr=condition['lr'], seed=0,
                                    parameter_count=sum(p.numel() for p in model.parameters()), windows=[])
            replay_previous = None
            replay_checks = []
            state = 0
            if condition['reconstructed']:
                initial = torch.load(ROOT/condition['initial'], weights_only=True)
                model.load_state_dict(initial)
                assert tensor_hash(model.parameters()) == condition['initial_tensor_sha256']
            for anchor in cfg['anchors']:
                if not condition['reconstructed']:
                    parent = next(p for p in condition['parents'] if p['state'] == anchor)
                    checkpoint = torch.load(ROOT/parent['path'], map_location='cpu', weights_only=False)
                    assert checkpoint['step'] == anchor
                    model.load_state_dict(checkpoint['model'])
                    assert tensor_hash(model.parameters()) == parent['tensor_sha256']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    torch.set_rng_state(checkpoint['torch_rng'].cpu())
                    if checkpoint['cuda_rng']:
                        torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint['cuda_rng']])
                    state = anchor
                    del checkpoint
                else:
                    while state < anchor:
                        loss, gradient = task.gradient()
                        raw = gradient.double()
                        row = dict(state=state, loss=loss, norm=float(raw.norm()), adjacent_cosine=None if replay_previous is None else cosine(replay_previous, raw))
                        if state in reference_rows:
                            replay_checks.append(qualify(row, reference_rows[state], cfg['qualification_tolerances']))
                        replay_previous = raw
                        optimizer.step()
                        state += 1
                        if state % 250 == 0:
                            event(phase='replay', parent=rid, completed_updates=state, target=anchor, loss=loss)
                initial_hash = tensor_hash(model.parameters())
                window = out/f'state-{anchor:06d}'
                window.mkdir()
                torch.save(dict(step=anchor, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), window/'start.pt')
                samples = np.lib.format.open_memmap(window/'gradients.npy', mode='w+', dtype=np.float32,
                    shape=(cfg['max_lag']+1,condition_result['parameter_count']))
                rows, checks, profile = [], [], []
                anchor_vector = previous = None
                elapsed_start = time.perf_counter()
                fixed_repeat = None
                for offset in range(cfg['max_lag']+1):
                    begin = time.perf_counter()
                    loss, gradient = task.gradient()
                    raw = gradient.double()
                    if not bool(torch.isfinite(raw).all()):
                        raise RuntimeError('Nonfinite gradient')
                    if offset == 0:
                        # Check that diagnostic recomputation leaves this full objective unchanged.
                        repeated_loss, repeated = task.gradient()
                        error = float((gradient-repeated).double().norm())
                        fixed_repeat = dict(loss_error=repeated_loss-loss, relative_gradient_error=error/max(float(raw.norm()),1e-30))
                        assert fixed_repeat['relative_gradient_error'] < cfg['fixed_repeat_relative_tolerance']
                        assert initial_hash == tensor_hash(model.parameters())
                        # The repeated backward owns p.grad and therefore the applied step.
                        loss, gradient = repeated_loss, repeated
                        raw = gradient.double()
                        anchor_vector = raw.clone()
                    samples[offset] = gradient.detach().cpu().numpy()
                    row = dict(state=anchor+offset, loss=loss, norm=float(raw.norm()),
                        adjacent_cosine=None if previous is None else cosine(previous, raw),
                        anchor_cosine=cosine(anchor_vector, raw))
                    rows.append(row)
                    if offset:
                        profile.append(dict(lag=offset, source_state=anchor, target_state=anchor+offset,
                                            cosine=row['anchor_cosine'], defined=row['anchor_cosine'] is not None))
                    if row['state'] in reference_rows:
                        checks.append(qualify(row, reference_rows[row['state']], cfg['qualification_tolerances']))
                    elif offset == 0 and anchor == 3000 and not condition['reconstructed']:
                        final_ok=close(loss,references['final_loss'],cfg['qualification_tolerances']['loss_absolute'],cfg['qualification_tolerances']['loss_relative'])
                        checks.append(dict(state=anchor, passed=final_ok,
                                           checks={'final_loss':final_ok}, loss_error=loss-references['final_loss']))
                    previous = raw
                    event(phase='gradient', parent=rid, task=task_name, anchor=anchor, offset=offset,
                          completed_updates=anchor+offset, loss=loss, norm=row['norm'], cosine=row['anchor_cosine'],
                          seconds=time.perf_counter()-begin, objective_units=task.n, loss_units=task.units,
                          supervised_tokens=task.units if task_name=='transformer' else None,
                          remaining_condition_gradients=(len(cfg['anchors'])-cfg['anchors'].index(anchor)-1)*(cfg['max_lag']+2)+cfg['max_lag']-offset)
                    if offset < cfg['max_lag']:
                        optimizer.step()
                samples.flush()
                del samples
                # At the final sampled state the last gradient was measured, not applied.
                state = anchor+cfg['max_lag']
                replay_previous = previous
                torch.save(dict(step=state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), window/'end.pt')
                historical_checks = replay_checks+checks if condition['reconstructed'] else checks
                qualified = bool(historical_checks) and all(c['passed'] for c in historical_checks)
                result = dict(anchor=anchor, last_state=state, gradients=len(rows), applied_updates=cfg['max_lag'],
                    parent_mode='reconstructed' if condition['reconstructed'] else 'saved_checkpoint',
                    starting_tensor_sha256=initial_hash, original_trajectory_qualified=qualified,
                    primary_eligible=qualified if condition['reconstructed'] else True,
                    interpretation='Short cloud continuation from exact saved model' if not condition['reconstructed'] else 'Historical reconstruction; eligibility requires every preceding replay check',
                    fixed_repeat=fixed_repeat, qualification=checks, preceding_replay_checks=len(replay_checks),
                    rows=rows, profile=profile, seconds=time.perf_counter()-elapsed_start,
                    extended_beyond_analysis024=anchor==3000, source_checkpoint=None if condition['reconstructed'] else parent)
                write_json(window/'result.json',result)
                condition_result['windows'].append(result)
                # Legacy replay continues at this state; a repeated fixed-state gradient
                # replaces previous consistently before the next actual update.
                if condition['reconstructed']:
                    replay_checks.extend(checks)
                    replay_previous = None
                del anchor_vector, previous, gradient, raw
            write_json(out/'replay_checks.json',replay_checks)
            write_json(out/'result.json',condition_result)
            completed.append(rid)
            event(phase='condition_complete', parent=rid, completed_conditions=len(completed))
            del task, model, optimizer, data
            torch.cuda.empty_cache()
        manifest.update(status='complete', completed_utc=stamp(), completed_conditions=completed,
                        elapsed_seconds=time.perf_counter()-clock)
        write_json(manifest_path,manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),completed_utc=stamp())
        write_json(manifest_path,manifest)
        raise
    finally:
        files=[]
        for path in sorted(args.output.rglob('*')):
            if path.is_file() and path.name != 'inventory.json':
                files.append(dict(path=path.relative_to(args.output).as_posix(),bytes=path.stat().st_size,sha256=file_hash(path)))
        write_json(args.output/'inventory.json',dict(files=files))


if __name__=='__main__':
    main()
