"""Execute the two-state continuation design with Run 214's measurement code."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import stamp, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', required=True, type=float)
    deadline = parser.parse_args().deadline
    path = ROOT / 'runs/214-2026-11-02-sgd-batch-extension/01_measure.py'
    spec = importlib.util.spec_from_file_location('measurement', path)
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    measurement.HERE = HERE
    config = json.loads((HERE / 'config.json').read_text())
    out = HERE / 'artifacts/results'
    out.mkdir(parents=True, exist_ok=True)
    for batch in config['batches']['transformer']:
        reps = 1 if batch == 59680 else config['stochastic_repetitions']
        for state in config['states']:
            for rep in range(reps):
                name = f'state-{state}-b{batch}-rep{rep:02d}'
                job = out / name
                remaining = deadline - time.time()
                if remaining < 15:
                    raise TimeoutError('Worker deadline reached')
                write_json(out / 'progress.json', dict(job=name, utc=stamp()))
                sys.argv = [str(path), '--mode', 'fork', '--task', 'transformer', '--seed', '0',
                            '--state', str(state), '--rep', str(rep), '--batch', str(batch),
                            '--reference', str(HERE / f'artifacts/state-{state:04}.pt'),
                            '--output', str(job), '--max-seconds', str(remaining)]
                measurement.main()
                assert (job / 'summary.json').is_file(), name
    assert len(list(out.glob('state-*/summary.json'))) == 66
    write_json(out / 'complete.json', dict(complete=66, utc=stamp()))

if __name__ == '__main__':
    main()
