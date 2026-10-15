"""Detached monitor, retrieval, plotting and report build for the owned sweep."""
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


def remote_probe(cohort, statefile):
    # Read only persisted progress; importing torch would allocate a CUDA
    # context and interfere with the active training process.
    return '''import json,pathlib,datetime
root=pathlib.Path(ROOT_VALUE)
cohort=json.load(open(root/'analyses/021-2026-10-15-eos-multitask-lr/cohort.json'))
state=json.load(open(STATE_VALUE))
active,plan,base=state,cohort,root
if state['status']=='waiting_for_predecessor':
    p=pathlib.Path(cohort['predecessor_state'])
    active=json.load(open(p));plan=json.load(open(p.parent/'cohort.json'));base=p.parents[2]
event=None
run=next((r for r in plan['runs'] if r['run_id']==active.get('current_run')),None)
if run:
    f=base/'runs'/run['run']/'artifacts/attempt-01-spark/production/events.jsonl'
    if f.exists() and f.stat().st_size:
        event=json.loads(f.read_text().splitlines()[-1])
def remaining(c,s,ev):
    total=0.
    for r in c['runs']:
        if r['kind']!='new_training':continue
        value=s['runs'].get(r['run_id'],{})
        if value.get('status')=='complete':continue
        seconds=r['estimated_seconds']
        if r['run_id']==s.get('current_run') and ev:
            seconds=ev['remaining_seconds_at_observed_rate']
        total+=seconds
    return total
seconds=remaining(plan,active,event)
if plan is not cohort:seconds+=remaining(cohort,state,None)
warnings=[]
if event and (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(event['utc'])).total_seconds()>900:
    warnings.append('No persisted update progress for more than15min; inspect Hessian phase/log.')
print(json.dumps({'state':state,'active_queue_run':active.get('current_run'),'progress':event,
    'remaining_expected_hours_including_queue':seconds/3600,'planning_allowance_hours':seconds*1.4/3600,'warnings':warnings}))
'''.replace('ROOT_VALUE', repr(cohort['remote_root'])).replace('STATE_VALUE', repr(statefile))


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    statefile = cohort['remote_root'] + '/analyses/' + HERE.name + '/queue_state.json'
    started = time.monotonic()
    while time.monotonic() - started < cohort['wait_max_seconds'] + cohort['hard_queue_seconds'] + 7200:
        try:
            code = remote_probe(cohort, statefile)
            remote = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'tcml-spark@spark-2277.local',
                shlex.join([cohort['remote_python'], '-c', code])]
            payload = json.loads(subprocess.check_output(remote, text=True, timeout=35))
            state = payload.pop('state')
            event = dict(utc=stamp(), status=state['status'], current_run=state.get('current_run'),
                completed_new_runs=sum(r['status'] == 'complete' for r in state['runs'].values()), total_new_runs=16,
                **payload)
            write_json(HERE / 'local-controller.json', event)
            with (HERE / 'monitoring.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event) + '\n')
        except Exception as error:
            write_json(HERE / 'local-controller.json', dict(status='read_retry', utc=stamp(), error=repr(error)))
            time.sleep(300)
            continue
        if state['status'] == 'complete':
            try:
                for script in ['03_collect.py', '04_plot.py', '05_manuscript.py']:
                    subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, check=True)
            except Exception as error:
                # Do not repeatedly re-download or overwrite a concurrently
                # edited report after a scientific/postprocessing failure.
                write_json(HERE / 'local-controller.json', dict(status='postprocessing_requires_review', utc=stamp(), error=repr(error)))
                return
            write_json(HERE / 'local-controller.json', dict(status='figure_and_report_generated_awaiting_visual_review', utc=stamp(),
                note='All evidence retrieved and checked; figure placed after Table1. Current renders need visual review before acceptance/commit.'))
            return
        if state['status'] in ['finished_with_failures', 'predecessor_wait_deadline']:
            write_json(HERE / 'local-controller.json', dict(status='requires_failure_review', utc=stamp(), remote_status=state))
            return
        time.sleep(300)
    write_json(HERE / 'local-controller.json', dict(status='monitor_deadline', utc=stamp()))


if __name__ == '__main__':
    main()
