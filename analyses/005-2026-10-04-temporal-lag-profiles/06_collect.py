"""Monitor only this cohort, then retrieve and verify before MCP teardown."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('operate', HERE / '02_operate.py')
operate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operate)
sys.stdout.reconfigure(encoding='utf-8')


def collect(row):
    pre = ROOT / 'runs' / row['run'] / 'prelaunch'
    operate.work(row, 'download')
    result = subprocess.run([sys.executable, str(HERE / '04_accept.py'), row['run']],
                            capture_output=True, text=True, encoding='utf-8', timeout=1200)
    (pre / '04_accept-log.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] + result.stdout[-1200:])
    print(json.dumps({'run': row['run'][:3], 'stage': 'accepted; ready for owned-worker deletion'}), flush=True)


def status(row):
    pre = ROOT / 'runs' / row['run'] / 'prelaunch'
    lease = json.loads((pre / 'lease.json').read_text())
    attempt = lease['attempt']
    if (pre / f'{attempt}-retrieval.json').exists():
        return {'run': row['run'][:3], 'stage': 'accepted'}
    snapshot = json.loads(operate.native(row, 'status'))
    manifest, event = snapshot.get('manifest', {}), snapshot.get('latest_update', {})
    result = {'run': row['run'][:3], 'stage': manifest.get('status'),
              'update': manifest.get('completed_updates'), 'input_tokens': manifest.get('completed_input_tokens'),
              'loss': manifest.get('latest_loss'), 'tokens_per_second': event.get('tokens_per_second'),
              'remaining_seconds': manifest.get('remaining_training_seconds'),
              'peak_memory_gb': event.get('peak_cuda_allocated_bytes', 0) / 1e9,
              'overflows': event.get('overflow_retries')}
    if manifest.get('status') == 'failed':
        raise RuntimeError(row['run'] + ': ' + manifest.get('error', 'training failed'))
    if snapshot.get('verification', {}).get('verified'):
        receipt = pre / f'archive-{attempt}-start.json'
        if not receipt.exists():
            operate.work(row, 'archive')
            receipt.write_text(json.dumps({'pod_id': lease['pod_id'],
                              'started_at': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        stem = '/opt/run' + row['run'][:3] + '-' + attempt + '.tar.gz'
        archive = operate.native(row, 'command',
                  'tail -n 3 /workspace/constructive/archive.log; cat ' + stem + '.sha256 ' + stem + '.bytes 2>/dev/null || true')
        (pre / f'archive-{attempt}-status.txt').write_text(archive)
        result['stage'] = 'archive-ready' if 'ARCHIVE_READY' in archive else 'archiving'
    return result


def main():
    rows = json.loads((HERE / 'cohort.json').read_text())
    if len(sys.argv)>1:
        rows=[row for row in rows if row['run'][:3] in sys.argv[1:]]
    pending = {}
    with ThreadPoolExecutor(max_workers=3) as monitors, ThreadPoolExecutor(max_workers=2) as transfers:
        while True:
            for run, future in list(pending.items()):
                if future.done():
                    future.result()
                    del pending[run]
            snapshots = list(monitors.map(status, rows))
            record = {'at': datetime.now(timezone.utc).isoformat(), 'workers': snapshots}
            with (HERE / 'monitoring.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            for row, snapshot in zip(rows, snapshots):
                if snapshot['stage'] == 'archive-ready' and row['run'] not in pending:
                    pending[row['run']] = transfers.submit(collect, row)
            if all(s['stage'] == 'accepted' for s in snapshots):
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
