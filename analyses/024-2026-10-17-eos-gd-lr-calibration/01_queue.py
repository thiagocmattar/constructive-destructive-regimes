"""Admit only the five authorized new runs; never resume a previous queue."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import stamp, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    cohort = json.loads((HERE / 'cohort.json').read_text())
    spec = importlib.util.spec_from_file_location('qualified_executor', ROOT / 'analyses/023-2026-10-15-eos-smallest-lr-calibration/01_queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.verify_sources(ROOT, HERE.name)
    records = [r for r in cohort['runs'] if r['kind'] == 'new_training']
    assert [r['run_id'] for r in records] == cohort['allowed_new_run_ids'] == ['122', '123', '124', '125', '126']
    assert cohort['resume_predecessor'] is False and cohort['hard_queue_seconds'] == 120 * 3600
    assert all(r['updates'] == 3000 and r['seed'] == 0 and r['method'] == 'gd' for r in records)
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not gpu, f'GPU occupied; no work admitted: {gpu}'
    if args.verify_only:
        write_json(HERE / 'remote_preflight.json', dict(status='passed', utc=stamp(), scientific_updates=0,
            allowed_run_ids=cohort['allowed_new_run_ids'], gpu_compute_processes=gpu, sources_verified=True,
            resume_predecessor=False))
        print('PASS: source/config hashes, five allowed IDs, fixed horizon, idle GPU, no predecessor continuation.')
        return
    status = HERE / 'queue_state.json'
    assert not status.exists(), 'Queue already executed; preserve its state.'
    try:
        queue.execute(ROOT, cohort, records, status, dict(runs={}, resume_predecessor=False), cohort['hard_queue_seconds'])
    finally:
        write_json(HERE / 'finish_only_status.json', dict(status='no_follow_up_training', utc=stamp(),
            resume_predecessor=False, allowed_run_ids=cohort['allowed_new_run_ids']))


if __name__ == '__main__': main()
