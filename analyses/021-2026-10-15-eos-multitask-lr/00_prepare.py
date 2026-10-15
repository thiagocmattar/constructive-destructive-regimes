"""Freeze only the extra GD learning rates, reusing matched completed prefixes."""
import copy
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

PREVIOUS = ROOT / 'analyses/020-2026-10-14-eos-multitask'
RATES = {'mlp': [.004, .01, .02], 'resnet': [.1, .2, .4], 'transformer': [.05, .1, .2]}


def main():
    assert not (HERE / 'cohort.json').exists(), 'Cohort already frozen.'
    old = json.loads((PREVIOUS / 'cohort.json').read_text())
    for item in json.loads((PREVIOUS / 'source_inventory.json').read_text())['files']:
        if item['path'].startswith('src/'):
            assert file_hash(ROOT / item['path']) == item['sha256'], 'Training code changed since qualification.'
    for name in ['mathematical_verification.json', 'smoke_verification.json']:
        assert json.loads((PREVIOUS / name).read_text())['status'] == 'passed'
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    records, files = [], []
    number = 104
    # Sustained complete-run timings (including spectra/checkpointing), not
    # an optimistic extrapolation from a backward-only benchmark.
    seconds = {'mlp': 12., 'resnet': 7220., 'transformer': 16010.}
    for task, rates in RATES.items():
        for lr in rates:
            for seed in [0, 1, 2]:
                baseline = next(r for r in old['runs'] if r['task'] == task and r['method'] == 'gd' and r['seed'] == seed)
                if lr == rates[1]:
                    record = copy.deepcopy(baseline)
                    record.update(kind='reused_gd_prefix', origin_analysis='020')
                elif task == 'mlp' and seed == 0:
                    run_id = '059' if lr == .004 else '062'
                    run = next((ROOT / 'runs').glob(run_id + '-*'))
                    cfg = json.loads((run / 'config.json').read_text())
                    manifest = json.loads((run / 'artifacts/attempt-01-spark/production/manifest.json').read_text())
                    assert cfg['optimizers']['gd']['lr'] == lr and cfg['selected_optimizer'] == 'gd'
                    assert manifest['status'] == 'complete' and manifest['completed_updates'] >= 500
                    assert manifest['initial_tensor_sha256'] == inputs['initial']['mlp'][seed]['initial_tensor_sha256']
                    record = dict(run_id=run_id, run=run.name, task=task, method='gd', seed=seed,
                        lr=lr, batch_units=None, updates=500, kind='reused_gd_prefix', origin_analysis='019')
                else:
                    cfg = copy.deepcopy(baseline['config'])
                    cfg.update(run_id=f'{number:03}', hypothesis='H020', lr=lr)
                    cfg['notes'] += ['Analysis021: matched LR ablation; three seeds, 500 fixed updates.']
                    name = f'{number:03}-2026-10-15-eos-{task}-gd-lr-{lr:g}-seed{seed}'
                    folder = ROOT / 'runs' / name
                    folder.mkdir(exist_ok=False)
                    write_json(folder / 'config.json', cfg)
                    (folder / 'README.md').write_bytes((f'# Run{number:03}: {task} full-batch GD\n\n'
                        f'LR{lr:g}, seed{seed}, 500 fixed updates on owned Spark.\n'
                        'The [Analysis021 protocol](../../analyses/' + HERE.name + '/PROTOCOL.md)\n'
                        'defines matched factors, measurements and limits. Executed scientific inputs\n'
                        'are frozen; status is in the attempt manifest.\n').encode())
                    files.append(folder / 'config.json')
                    record = dict(run_id=f'{number:03}', run=name, kind='new_training', task=task,
                        method='gd', seed=seed, lr=lr, batch_units=None, updates=500,
                        estimated_seconds=seconds[task], maximum_seconds=int(seconds[task] * 1.6 + 900),
                        config=cfg, config_sha256=file_hash(folder / 'config.json'))
                    number += 1
                records.append(record)
    expected = sum(r.get('estimated_seconds', 0) for r in records if r['kind'] == 'new_training') / 3600
    remote = '/home/tcml-spark/experiments/constructive-destructive-regimes/eos-multitask-lr-2026-10-15'
    cohort = {k: old[k] for k in ['remote_python', 'remote_dependencies', 'remote_cache', 'remote_mlp_cache']}
    cohort.update(analysis_id='021', hypothesis='H020', created_utc=stamp(), remote_root=remote,
        predecessor_state=old['remote_root'] + '/analyses/' + PREVIOUS.name + '/queue_state.json',
        wait_max_seconds=72 * 3600, hard_queue_seconds=72 * 3600, estimated_hours=expected,
        qualified_hours=expected * 1.4, owned_spark=True, rental_cost=0,
        additional_storage_allowance_bytes=25_000_000_000, horizon='matched-first-500',
        runs=records, next_run=number, next_analysis=22, seeds=[0, 1, 2], rates=RATES)
    assert len(records) == 27 and number == 120
    write_json(HERE / 'cohort.json', cohort)
    write_json(HERE / 'preflight.json', {'status': 'passed', 'utc': stamp(),
        'checks': ['Analysis020 mathematical and full-shape smoke checks passed; unchanged training code',
            '27 matched conditions; 11 reused and 16 new; seeds0/1/2 at every rate',
            'Run059/062 seed0 tensors match the prepared MLP initialization',
            'Earlier queue retained; isolated remote root; no simultaneous GPU runs'],
        'references': {n: file_hash(PREVIOUS / n) for n in ['mathematical_verification.json', 'smoke_verification.json']}})
    files += list((ROOT / 'src/eos_tasks').glob('*.py')) + list((ROOT / 'src/cifar_eos').glob('*.py'))
    files += [HERE / n for n in ['cohort.json', 'preflight.json', '01_queue.py']]
    inventory = {'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in files]}
    write_json(HERE / 'source_inventory.json', inventory)
    packet = ROOT / 'cache/eos-lr-sources.tar'
    with tarfile.open(packet, 'w') as bundle:
        for path in files + [HERE / 'source_inventory.json']:
            bundle.add(path, arcname=path.relative_to(ROOT).as_posix())
    write_json(HERE / 'packet.json', {'path': packet.relative_to(ROOT).as_posix(), 'bytes': packet.stat().st_size, 'sha256': file_hash(packet)})
    print(json.dumps({'new_runs': 16, 'reused': 11, 'estimated_hours': expected, 'qualified_hours': expected * 1.4, 'next_run': number}))


if __name__ == '__main__':
    main()
