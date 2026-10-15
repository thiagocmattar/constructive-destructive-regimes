"""Detached local monitor, automatic verified retrieval and numeric analysis."""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import stamp, write_json


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    remote = cohort['remote_root'] + '/analyses/' + HERE.name + '/queue_state.json'
    started = time.monotonic()
    while time.monotonic()-started < cohort['drain_max_seconds']+cohort['hard_queue_seconds']+7200:
        try:
            code = 'import pathlib,json; p=pathlib.Path('+repr(remote)+'); s=json.loads(p.read_text()); print(json.dumps(s))'
            args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'tcml-spark@spark-2277.local',
                    shlex.join([cohort['remote_python'], '-c', code])]
            state = json.loads(subprocess.check_output(args, text=True, timeout=30))
            event = dict(utc=stamp(), status=state['status'], current_run=state.get('current_run'),
                predecessor_run=state.get('predecessor_run'),
                completed_new_runs=sum(r['status']=='complete' for r in state['runs'].values()),
                total_new_runs=2)
            write_json(HERE / 'local-controller.json', event)
            with (HERE / 'monitoring.jsonl').open('a') as out: out.write(json.dumps(event)+'\n')
            if state['status'] == 'complete':
                for script in ['04_collect.py', '05_reduce.py']:
                    subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, check=True)
                write_json(HERE / 'local-controller.json', dict(status='retrieved_and_numerically_verified', utc=stamp(),
                    manuscript_edits=False, results='results.json', note='No visual acceptance or manuscript changes performed.'))
                return
            if state['status'] == 'finished_with_failures':
                write_json(HERE / 'local-controller.json', dict(status='requires_failure_review', utc=stamp(), queue=state))
                return
        except Exception as error:
            write_json(HERE / 'local-controller.json', dict(status='retrying_read_or_requires_review', utc=stamp(), error=repr(error)))
        time.sleep(300)
    write_json(HERE / 'local-controller.json', dict(status='controller_deadline', utc=stamp()))


if __name__ == '__main__':
    main()
