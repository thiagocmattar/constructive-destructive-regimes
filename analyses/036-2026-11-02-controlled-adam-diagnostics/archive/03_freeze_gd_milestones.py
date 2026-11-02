"""Freeze historical GD training-loss targets before scientific Adam screening."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json

HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'analyses/024-2026-10-17-eos-gd-lr-calibration'


def main():
    cohort = json.loads((SOURCE/'cohort.json').read_text())
    results = json.loads((SOURCE/'results.json').read_text())
    assert results['status'] == 'passed' and results['available'] == 9
    targets = []
    for record in cohort['runs']:
        prior = next(row for row in results['runs'] if row['run_id'] == record['run_id'])
        folder = ROOT/'runs'/record['run']/'artifacts/attempt-01-spark/production'
        for filename, sha in prior['source_sha256'].items():
            assert file_hash(folder/filename) == sha
        rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        legacy = record['kind'] == 'reused_gd_prefix'
        def loss_at(state):
            if state < len(rows):
                row = rows[state]
                assert row['update'] == state+1
                return row['train_loss'] if legacy else row['full_objective_loss']
            assert state == 3000
            return json.loads((folder/'summary.json').read_text())['final_train']['loss']
        losses = {str(state): loss_at(state) for state in [500, 1000, 2000, 3000]}
        assert all(value is not None for value in losses.values())
        assert abs(losses['3000']-prior['final_loss']) < 1e-12
        targets.append(dict(run_id=record['run_id'], task=record['task'], gd_lr=record['lr'],
            seed=0, targets_by_parameter_state=losses, source=folder.relative_to(ROOT).as_posix(),
            source_sha256=prior['source_sha256'],
            summary_sha256=file_hash(folder/'summary.json')))
    write_json(HERE/'gd_loss_milestones.json', dict(status='frozen_before_scientific_adam_screening',
        analysis024_results_sha256=file_hash(SOURCE/'results.json'),
        analysis024_cohort_sha256=file_hash(SOURCE/'cohort.json'), conditions=targets,
        reduction='All nine GD conditions; exact full-objective losses at parameter states500/1000/2000/3000.',
        comparison='First Adam state and first64-state mean at/below each target; right censor unachieved targets.',
        limits='One seed; same loss need not mean same region. Historical hardware/optimizer costs differ.'))
    print(json.dumps(dict(conditions=len(targets), frozen_targets=4*len(targets))))


if __name__ == '__main__':
    main()
