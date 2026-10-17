"""Hidden 30-minute reads and terminal retrieval; never admits training."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import stamp, write_json
from importlib import import_module

POLL_SECONDS = 1800
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def main():
    helper = import_module('02_launch')
    c = json.loads((HERE / 'cohort.json').read_text())
    start = time.monotonic()
    processed = set()
    while time.monotonic()-start < c['hard_queue_seconds']+4*3600:
        try:
            code = f'''import pathlib,json,subprocess
r=pathlib.Path({c['remote_root']!r})
s=json.loads((r/'analyses'/{HERE.name!r}/'queue_state.json').read_text())
event=None
if s.get('current_run'):
 p=next((r/'runs').glob(s['current_run']+'-*'))/'artifacts/attempt-01-spark/production/events.jsonl'
 if p.exists() and p.read_text().strip(): event=json.loads(p.read_text().splitlines()[-1])
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
print(json.dumps(dict(queue=s,progress=event,gpu_util_temp_power=gpu)))'''
            snapshot = json.loads(helper.remote(c['remote_python'], code))
            state = snapshot['queue']
            completed = {rid for rid, r in state['runs'].items() if r['status'] == 'complete'}
            event = dict(utc=stamp(), status=state['status'], current_run=state.get('current_run'),
                completed_new_runs=len(completed), total_new_runs=5, poll_seconds=POLL_SECONDS,
                progress=snapshot['progress'], gpu_util_temp_power=snapshot['gpu_util_temp_power'])
            progress = snapshot['progress']
            pending = [r for r in c['runs'] if r['kind'] == 'new_training' and r['run_id'] not in state['runs']]
            current_remaining = progress['remaining_seconds_at_observed_rate'] if progress else next(
                (r['estimated_seconds'] for r in c['runs'] if r['run_id'] == state.get('current_run')), 0)
            event['estimated_queue_remaining_hours'] = (sum(r['estimated_seconds'] for r in pending)
                + current_remaining)/3600
            write_json(HERE / 'local-controller.json', event)
            with (HERE / 'monitoring.jsonl').open('a') as out: out.write(json.dumps(event)+'\n')
            if completed-processed or state['status'] in ['complete', 'finished_with_failures']:
                subprocess.run([sys.executable, str(HERE / '03_results.py'), '--collect'], cwd=ROOT,
                    check=True, creationflags=FLAGS)
                processed = completed
            if state['status'] in ['complete', 'finished_with_failures']:
                write_json(HERE / 'local-controller.json', dict(status='retrieved_and_numerically_verified' if state['status'] == 'complete'
                    else 'requires_failure_review', utc=stamp(), queue=state, no_follow_up_training=True,
                    manuscript_edits=False, figures='Final comparison awaits calibration review.'))
                return
        except Exception as error:
            write_json(HERE / 'local-controller.json', dict(status='retrying_read_or_requires_review', utc=stamp(), error=repr(error)))
        time.sleep(POLL_SECONDS)
    write_json(HERE / 'local-controller.json', dict(status='monitor_deadline', utc=stamp()))


if __name__ == '__main__': main()
