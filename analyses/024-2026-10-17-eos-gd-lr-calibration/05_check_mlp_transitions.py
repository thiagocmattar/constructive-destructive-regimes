"""Separate initial sign reversals from persistent negative alignment in MLP pilots."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json


def main():
    records = json.loads((HERE / 'cohort.json').read_text())['runs']
    results = []
    for r in records:
        if r['task'] != 'mlp': continue
        metrics = ROOT / 'runs' / r['run'] / 'artifacts/attempt-01-spark/production/metrics.jsonl'
        rows = [json.loads(s) for s in metrics.read_text().splitlines()][:3000]
        c = np.asarray([np.nan if x['streams']['raw']['cosine'] is None else x['streams']['raw']['cosine'] for x in rows])
        first = np.flatnonzero(c < 0)
        windows = {}
        for fraction in [.8, 1.0]:
            ends = [i+1 for i in range(100, len(c)) if np.mean(c[i-99:i+1] < 0) >= fraction]
            windows[str(fraction)] = ends[0] if ends else None
        results.append(dict(run_id=r['run_id'], lr=r['lr'], updates=len(rows),
            first_negative_gradient_update=int(first[0]+1) if len(first) else None,
            first_100_pair_window_end_update=windows, metrics_sha256=file_hash(metrics)))
    write_json(HERE / 'mlp_transition_check.json', dict(status='passed', runs=results,
        definition='Window comprises 100 defined adjacent pairs, inclusive; report its ending gradient update (one-based). Threshold0.8 matches Analysis023 descriptive convention. A window threshold is not a stability test.',
        script_sha256=file_hash(__file__)))
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
