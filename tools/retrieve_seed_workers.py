"""Wait for owned archives, then transfer and accept each exactly once."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
cohort = json.loads((ROOT / 'cache/runpod/seed-cohort.json').read_text())
chosen = [r for r in cohort if r['run'][:3] in sys.argv[1:]]
assert len(chosen) == len(set(sys.argv[1:])) and chosen

def work(item):
    folder = item['run']; run = ROOT / 'runs' / folder
    assert not (run / 'prelaunch/attempt1-retrieval.json').exists()
    end = time.monotonic() + 600
    while True:
        status = subprocess.run([sys.executable, str(ROOT / 'tools/scale_worker.py'), folder, 'archive-status'],
            capture_output=True, text=True, encoding='utf-8', timeout=90)
        assert status.returncode == 0, status.stderr
        # The exact bytes and digest are independently re-read by the transfer helper.
        if 'ARCHIVE_READY' in status.stdout and '.tar.gz' in status.stdout:
            print(json.dumps({'run':folder[:3], 'stage':'archive_ready'}), flush=True)
            break
        assert time.monotonic() < end, 'Packaging deadline reached; preserve worker and investigate.'
        print(json.dumps({'run':folder[:3], 'stage':'packaging'}), flush=True)
        time.sleep(30)
    for script, stage in [('download_scale_archive.py','transfer'), ('accept_scale_archive.py','local_verification')]:
        print(json.dumps({'run':folder[:3], 'stage':stage}), flush=True)
        subprocess.run([sys.executable, str(ROOT / 'tools' / script), folder], check=True)
    print(json.dumps({'run':folder[:3], 'stage':'accepted_ready_for_cleanup'}), flush=True)

with ThreadPoolExecutor(max_workers=min(2, len(chosen))) as pool:
    list(pool.map(work, chosen))
