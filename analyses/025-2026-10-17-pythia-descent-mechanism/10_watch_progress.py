"""Refresh progress independently of blocking checkpoint downloads; no mutations."""
from pathlib import Path
import importlib.util
import json
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('h021_control', HERE / '04_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def read_optional(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def main():
    lease = json.loads((HERE / 'lease.json').read_text())
    assert lease['owner'] == 'Analysis025-current-task'
    while time.time() < lease['hard_deadline_epoch']:
        if (HERE / 'closeout.json').exists():
            return
        idle = read_optional(HERE / 'idle-stop-status.json', {})
        record = dict(observed_at=time.time(), verified_runs=len(
            read_optional(HERE / 'retrieved.json', {'runs': []})['runs']),
            budget=idle.get('budget'), pods=[])
        for pod in lease['pods']:
            stopped = idle.get('pods', {}).get(pod['id'], {})
            if stopped.get('runtimeStatus') in {'stopped', 'terminated'}:
                record['pods'].append(dict(stopped, pod=pod['id'], status='stopped_after_verification'))
                continue
            try:
                record['pods'].append(dict(pod=pod['id'], **control.remote_status(pod, lease)))
            except Exception as error:
                record['pods'].append(dict(pod=pod['id'], observed_at=time.time(), error=repr(error)))
        record['completed_at_epoch'] = time.time()
        control.save(HERE / 'fresh-progress.json', record)
        print(json.dumps(dict(observed_at=record['observed_at'], verified_runs=record['verified_runs'],
            pods=[dict(pod=r['pod'], status=r.get('status'),
                update=r.get('latest_update', {}).get('update'), error=r.get('error'))
                for r in record['pods']])), flush=True)
        time.sleep(300)


if __name__ == '__main__':
    main()
