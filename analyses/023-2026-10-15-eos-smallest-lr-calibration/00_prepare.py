"""Freeze the requested single-seed, smallest-LR 3k Spark calibration."""
import copy
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    assert not (HERE / 'cohort.json').exists(), 'Inputs already frozen.'
    previous = ROOT / 'analyses/020-2026-10-14-eos-multitask'
    old = json.loads((previous / 'cohort.json').read_text())
    files = []
    for item in json.loads((previous / 'source_inventory.json').read_text())['files']:
        if item['path'].startswith('src/'):
            assert file_hash(ROOT / item['path']) == item['sha256'], item['path']
    for name in ['mathematical_verification.json', 'smoke_verification.json']:
        assert json.loads((previous / name).read_text())['status'] == 'passed'
    records = []
    run = next((ROOT / 'runs').glob('059-*'))
    cfg = json.loads((run / 'config.json').read_text())
    assert cfg['selected_optimizer'] == 'gd' and cfg['optimizers']['gd']['lr'] == .004
    manifest = json.loads((run / 'artifacts/attempt-01-spark/production/manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed_updates'] >= 3000
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    assert manifest['initial_tensor_sha256'] == inputs['initial']['mlp'][0]['initial_tensor_sha256']
    records.append(dict(run_id='059', run=run.name, kind='reused_gd_prefix', task='mlp',
                        method='gd', seed=0, lr=.004, updates=3000, origin_analysis='019'))
    for number, task, lr, estimated in [(120, 'resnet', .1, 12.01 * 3600),
                                        (121, 'transformer', .05, 26.69 * 3600)]:
        baseline = next(r for r in old['runs'] if r['task'] == task and r['method'] == 'gd' and r['seed'] == 0)
        cfg = copy.deepcopy(baseline['config'])
        cfg.update(run_id=f'{number:03}', hypothesis='H020', lr=lr, updates=3000,
                   horizon='smallest-lr-3000-calibration', retained_states=[500, 1000, 2000, 3000])
        cfg['notes'] += ['Analysis023: user-requested smallest-LR seed0 calibration, 3000 fixed full-GD updates.']
        name = f'{number:03}-2026-10-15-eos-{task}-gd-smallest-lr-3k'
        folder = ROOT / 'runs' / name
        folder.mkdir(exist_ok=False)
        write_json(folder / 'config.json', cfg)
        (folder / 'README.md').write_bytes((f'# Run{number}: {task}, smallest-LR 3k calibration\n\n'
            f'Full-batch GD, LR{lr:g}, seed0, 3,000 fixed updates on owned Spark.\n'
            f'[Analysis023 protocol](../../analyses/{HERE.name}/README.md) defines scope,\n'
            'diagnostics, retention, priority scheduling and limitations. No cloud rental.\n'
            'New horizon uses a new ID; old executed configs remain unchanged.\n').encode())
        files.append(folder / 'config.json')
        records.append(dict(run_id=f'{number:03}', run=name, kind='new_training', task=task,
            method='gd', seed=0, lr=lr, batch_units=None, updates=3000,
            estimated_seconds=estimated, maximum_seconds=int(estimated * 1.5 + 1800),
            config=cfg, config_sha256=file_hash(folder / 'config.json')))
    cohort = {k: old[k] for k in ['remote_python', 'remote_dependencies', 'remote_cache', 'remote_mlp_cache']}
    launch = json.loads((previous / 'training_launch.json').read_text())
    cohort.update(analysis_id='023', hypothesis='H020', created_utc=stamp(),
        remote_root='/home/tcml-spark/experiments/constructive-destructive-regimes/eos-smallest-lr-3k-2026-10-15',
        predecessor_root=old['remote_root'], predecessor_analysis=previous.name,
        predecessor_queue_pid=launch['pid'], drain_max_seconds=8 * 3600,
        hard_queue_seconds=60 * 3600, predecessor_continuation_seconds=72 * 3600,
        owned_spark=True, rental_cost=0, additional_storage_allowance_bytes=12_000_000_000,
        seeds=[0], runs=records, estimated_hours=38.70, qualified_hours=54.18,
        next_run=122, next_analysis=24)
    write_json(HERE / 'cohort.json', cohort)
    write_json(HERE / 'preflight.json', dict(status='passed', utc=stamp(),
        checks=['Qualified scientific code unchanged since Analysis020',
                'Seed0 for all cases; smallest LRs .004/.1/.05; 3000 updates',
                'MLP reuses verified executed Run059 prefix, without rerunning',
                'New larger-model horizon has fresh IDs and exact frozen initialization',
                'Priority handoff drains the active run; unchanged pending020 jobs resume afterwards'],
        references={n: file_hash(previous / n) for n in ['mathematical_verification.json', 'smoke_verification.json']}))
    files += list((ROOT / 'src/eos_tasks').glob('*.py')) + list((ROOT / 'src/cifar_eos').glob('*.py'))
    files += [HERE / n for n in ['cohort.json', 'preflight.json', '01_queue.py', '03_verify_scheduling.py']]
    write_json(HERE / 'source_inventory.json', {'files': [dict(path=p.relative_to(ROOT).as_posix(),
        bytes=p.stat().st_size, sha256=file_hash(p)) for p in files]})
    packet = ROOT / 'cache/eos-smallest-lr-3k-sources.tar'
    with tarfile.open(packet, 'w') as bundle:
        for p in files + [HERE / 'source_inventory.json']:
            bundle.add(p, arcname=p.relative_to(ROOT).as_posix())
    write_json(HERE / 'packet.json', dict(path=packet.relative_to(ROOT).as_posix(),
        bytes=packet.stat().st_size, sha256=file_hash(packet)))
    print(json.dumps(dict(new_runs=2, reused=1, updates=3000, expected_hours=38.7, rental_cost=0)))


if __name__ == '__main__':
    main()
