"""Training-loss screening labels; thresholds are descriptive, not stability tests."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cifar_eos.core import file_hash


def classify(losses, status='complete', window=64):
    if status != 'complete' or len(losses) < window+1 or not all(math.isfinite(x) and x >= 0 for x in losses):
        return dict(label='incomplete_or_failed', status=status)
    tail = losses[-window:]
    changes = [b-a for a, b in zip(tail[:-1], tail[1:])]
    relative_changes = [abs(b-a)/max(abs(a), 1e-12) for a, b in zip(tail[:-1], tail[1:])]
    rises = sum(b-a > 1e-5*max(abs(a), 1e-12) for a, b in zip(tail[:-1], tail[1:])) / len(changes)
    terminal = statistics.mean(tail)
    median_change = statistics.median(relative_changes)
    progress = terminal <= .95*losses[0]
    smooth = progress and rises <= .05
    oscillatory = rises >= .25 and median_change >= .001 and terminal <= losses[0]
    label = 'smooth_progress' if smooth else 'oscillatory_progress' if oscillatory else 'unclassified'
    return dict(label=label, tail_window=window, upward_fraction=rises,
        median_relative_absolute_change=median_change, terminal_mean=terminal,
        terminal_over_initial=None if losses[0] == 0 else terminal/losses[0],
        tail_min=min(tail), tail_max=max(tail), peak_loss=max(losses),
        startup_peak_loss=max(losses[:window]), final_loss=losses[-1])


def select_regimes(rows):
    smooth = [r for r in rows if r['label'] == 'smooth_progress']
    oscillatory = [r for r in rows if r['label'] == 'oscillatory_progress']
    low = min(smooth, key=lambda r: r['lr'])['lr'] if smooth else None
    high = max(oscillatory, key=lambda r: r['lr'])['lr'] if oscillatory else None
    middle = None
    if low is not None and high is not None and high > low:
        candidates = [r['lr'] for r in rows if low < r['lr'] < high and r.get('status') == 'complete']
        if candidates:
            target = (math.log(low)+math.log(high))/2
            middle = min(candidates, key=lambda rate: (abs(math.log(rate)-target), rate))
    return dict(smooth_lr=low, intermediate_lr=middle, oscillatory_lr=high,
        spans_requested_regimes=low is not None and high is not None and low < high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in args.outputs:
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest.get('mechanics_only'):
            raise ValueError('Smoke checks cannot enter scientific calibration.')
        boundary_file = folder/'boundaries.jsonl'
        if boundary_file.exists():
            identity = manifest.get('files', {}).get('boundaries.jsonl')
            if identity is None or boundary_file.stat().st_size != identity['bytes'] or file_hash(boundary_file) != identity['sha256']:
                raise ValueError('Boundary artifact integrity failed.')
            boundaries = [json.loads(line) for line in boundary_file.read_text().splitlines()]
        else:
            boundaries = []
        if manifest['status'] == 'complete' and [r['state'] for r in boundaries] != list(range(manifest['effective_updates']+1)):
            raise ValueError('A complete zero-based boundary history is required; qualify and join resume fragments first.')
        losses = [row['loss'] for row in boundaries]
        result = classify(losses, manifest['status'])
        rows.append(dict(result, task=manifest['task'], lr=manifest['lr'], status=manifest['status'], source=str(folder)))
    keys = [(r['task'], r['lr']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate task/LR arms require canonical attempt qualification first.')
    config_path = Path(__file__).with_name('config.json')
    config = json.loads(config_path.read_text())
    expected = {(task, lr) for task in config['tasks'] for lr in config['learning_rates']}
    if not set(keys).issubset(expected):
        raise ValueError('An arm is outside the declared task/LR grid.')
    complete = set(keys) == expected
    result = dict(arms=rows, cohort_complete=complete,
        candidates={task: select_regimes([r for r in rows if r['task'] == task]) for task in config['tasks']} if complete else {},
        note='Training-only descriptive selection; no stability, convergence or optimizer-benefit claim.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
