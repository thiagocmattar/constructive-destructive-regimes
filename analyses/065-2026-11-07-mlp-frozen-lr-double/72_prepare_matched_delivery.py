"""Reduce and render the completed fleet; leave visual acceptance to the reviewer."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = HERE / 'scratch/h051'


def run(script, *arguments):
    print(json.dumps(dict(event='starting', script=script, arguments=arguments)), flush=True)
    subprocess.run([sys.executable, str(HERE/script), *arguments], cwd=ROOT, check=True)


def collect_case(case, workers):
    while True:
        leases = [SCRATCH/f'lease-{case}-{n}.json' for n in workers]
        if all(p.exists() for p in leases):
            records = [json.loads(p.read_text()) for p in leases]
            if any(p.get('budget_stopped') or p.get('stopped_epoch') for p in records):
                raise RuntimeError(f'{case}: bounded execution stopped; manual review required')
            if all(p.get('deleted_epoch') and p.get('retrieval_verified') for p in records):
                break
        time.sleep(20)
    run('63_retrieve_spark_matched.py', case)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(collect_case, 'resnet', [0, 1]),
                   pool.submit(collect_case, 'transformer', range(1, 7))]
        for future in futures:
            future.result()
    run('60_reduce_matched.py')
    run('64_verify_matched_reductions.py')
    run('71_check_matched_tracing.py')
    # The immutable prefetch may be finishing its last transfer. Cleanup refuses
    # to remove its task key until that process is absent.
    for attempt in range(3):
        try:
            run('70_close_matched_compute.py')
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(30)
    run('61_plot_matched.py')
    run('67_verify_matched_figures.py')
    print(json.dumps(dict(status='awaiting_visual_review',
                          figures=str(HERE/'figures/matched'))), flush=True)


if __name__ == '__main__':
    main()
