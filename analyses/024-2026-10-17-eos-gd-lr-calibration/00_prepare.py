"""Freeze the nine-condition, one-initialization, 3k GD learning-rate design."""
import copy
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    assert not (HERE / 'cohort.json').exists(), 'Scientific inputs are already frozen.'
    old = ROOT / 'analyses/023-2026-10-15-eos-smallest-lr-calibration'
    previous = json.loads((old / 'cohort.json').read_text())
    qualified = ROOT / 'analyses/020-2026-10-14-eos-multitask'
    for item in json.loads((qualified / 'source_inventory.json').read_text())['files']:
        if item['path'].startswith('src/'):
            assert file_hash(ROOT / item['path']) == item['sha256'], item['path']
    for name in ['mathematical_verification.json', 'smoke_verification.json']:
        assert json.loads((qualified / name).read_text())['status'] == 'passed'
    spec = importlib.util.spec_from_file_location('existing_checks', ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    records, files = [], []
    for rid in ['059', '062', '120', '121']:
        run = next((ROOT / 'runs').glob(rid + '-*'))
        cfg = json.loads((run / 'config.json').read_text())
        legacy = rid in ['059', '062']
        record = dict(run_id=rid, run=run.name, kind='reused_gd_prefix' if legacy else 'reused_completed_run',
            task='mlp' if legacy else cfg['task'], method='gd', seed=0, updates=3000,
            lr=cfg['optimizers']['gd']['lr'] if legacy else cfg['lr'], config_sha256=file_hash(run / 'config.json'))
        if legacy: checks.verify_legacy(record)
        else: assert checks.verified_local(run.name)
        records.append(record)
    seconds = {task: json.loads((ROOT / 'runs' / next(r['run'] for r in records if r['task'] == task)
        / 'artifacts/attempt-01-spark/production/summary.json').read_text())['elapsed_seconds']
        for task in ['resnet', 'transformer']}
    seconds['mlp'] = 180.0  # Conservative setup/31-spectrum allowance; legacy 20k runs took ~6min.
    for number, task, lr in [(122, 'mlp', .04), (123, 'resnet', .025),
                              (124, 'resnet', .05), (125, 'transformer', .1), (126, 'transformer', .2)]:
        baseline = next(r for r in previous['runs'] if r['task'] == ('resnet' if task == 'mlp' else task))
        cfg = copy.deepcopy(baseline['config'])
        cfg.update(run_id=str(number), task=task, lr=lr, horizon='single-initialization-lr-3000-calibration',
            hypothesis='H020', updates=3000, retained_states=[500, 1000, 2000, 3000])
        if task == 'mlp': cfg.update(physical=1000, hessian_units=5000)
        cfg['notes'] = cfg['notes'][:5] + ['Analysis024: seed0 matched across rates; 3000 full-GD updates; no old queue continuation.']
        name = f'{number}-2026-10-17-eos-{task}-gd-lr-{lr:g}-3k'
        folder = ROOT / 'runs' / name
        folder.mkdir(exist_ok=False)
        write_json(folder / 'config.json', cfg)
        (folder / 'README.md').write_text(f'# Run{number}: {task}, GD LR{lr:g}, 3k updates\n\n'
            f'User-authorized seed0 calibration on owned Spark. See [Analysis024](../../analyses/{HERE.name}/README.md)\n'
            'for matched factors, diagnostics, runtime bounds and retention. No cloud rental.\n', encoding='utf-8')
        files.append(folder / 'config.json')
        records.append(dict(run_id=str(number), run=name, kind='new_training', task=task, method='gd', seed=0,
            lr=lr, updates=3000, config=cfg, config_sha256=file_hash(folder / 'config.json'),
            estimated_seconds=seconds[task], maximum_seconds=3600 if task == 'mlp' else int(1.5 * seconds[task] + 1800)))
    expected = sum(r.get('estimated_seconds', 0) for r in records) / 3600
    cohort = {k: previous[k] for k in ['remote_python', 'remote_dependencies', 'remote_cache', 'remote_mlp_cache']}
    cohort.update(analysis_id='024', hypothesis='H020', created_utc=stamp(), runs=records, seeds=[0],
        remote_root='/home/tcml-spark/experiments/constructive-destructive-regimes/eos-gd-lr-3k-2026-10-17',
        hard_queue_seconds=120 * 3600, owned_spark=True, rental_cost=0,
        additional_storage_allowance_bytes=15_000_000_000, estimated_hours=expected, qualified_hours=1.2 * expected,
        resume_predecessor=False, allowed_new_run_ids=['122', '123', '124', '125', '126'], next_run=127, next_analysis=25)
    write_json(HERE / 'cohort.json', cohort)
    write_json(HERE / 'preflight.json', dict(status='passed', utc=stamp(),
        checks=['Scientific source hashes match qualified Analysis020', 'Four reused sources independently rehashed',
                'Nine conditions; three rates/case; same frozen seed0; 3000 full-GD updates',
                'Only five new IDs; old queues remain held; 120h hard queue limit'],
        qualifications={n: file_hash(qualified / n) for n in ['mathematical_verification.json', 'smoke_verification.json']}))
    files += list((ROOT / 'src/eos_tasks').glob('*.py')) + list((ROOT / 'src/cifar_eos').glob('*.py'))
    files += [old / '01_queue.py'] + list(HERE.glob('*.py')) + [HERE / 'cohort.json', HERE / 'preflight.json']
    write_json(HERE / 'source_inventory.json', {'files': [dict(path=p.relative_to(ROOT).as_posix(),
        bytes=p.stat().st_size, sha256=file_hash(p)) for p in files]})
    packet = ROOT / 'cache/eos-gd-lr-3k-2026-10-17-sources.tar'
    with tarfile.open(packet, 'w') as bundle:
        for p in files + [HERE / 'source_inventory.json']:
            bundle.add(p, arcname=p.relative_to(ROOT).as_posix())
    write_json(HERE / 'packet.json', dict(path=packet.relative_to(ROOT).as_posix(), bytes=packet.stat().st_size,
        sha256=file_hash(packet)))
    print(json.dumps(dict(status='prepared', conditions=9, new_runs=5, expected_hours=expected, qualified_hours=1.2 * expected)))


if __name__ == '__main__': main()
