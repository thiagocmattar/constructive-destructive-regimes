"""Replace admission-only processes while preserving the running optimizer child."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    launch = json.loads((HERE / 'launch.json').read_text())
    assert not (HERE / 'finish_only_policy.json').exists(), 'Override already applied.'
    source = HERE / '07_finish_only.py'
    remote = cohort['remote_root'] + '/analyses/' + HERE.name
    host = 'tcml-spark@spark-2277.local'
    subprocess.run(['scp', '-q', str(source), host+':'+remote+'/07_finish_only.py'], check=True)
    policy = dict(utc=stamp(), instruction="Let the active run and the two 3000-step calibrations finish. Do not launch anything afterwards; present the results before deciding next steps.",
        allowed_new_run_ids=['120', '121'], resume_predecessor=False,
        scientific_inputs_changed=False, replacement_script_sha256=file_hash(source),
        previous_coordinator_pid=launch['pid'], previous_admission_pid=cohort['predecessor_queue_pid'])
    code = '''import json,os,signal,time,subprocess,hashlib
from pathlib import Path
a=Path(ANALYSIS)
c=json.loads((a/'cohort.json').read_text())
policy=POLICY
script=a/'07_finish_only.py'
assert hashlib.file_digest(script.open('rb'),'sha256').hexdigest()==policy['replacement_script_sha256']
assert not (a/'finish_only_policy.json').exists()
def command(pid): return Path(f'/proc/{pid}/cmdline').read_bytes().replace(bytes([0]),b' ').decode()
def state(pid):
 p=Path(f'/proc/{pid}/stat')
 return p.read_text().rsplit(')',1)[1].split()[0] if p.exists() else None
coordinator=policy['previous_coordinator_pid']
admission=policy['previous_admission_pid']
assert str(a/'01_queue.py') in command(coordinator)
oldhere=Path(c['predecessor_root'])/'analyses'/c['predecessor_analysis']
assert str(oldhere/'03_queue.py') in command(admission)
assert state(admission) in ['T','t']
os.kill(coordinator,signal.SIGSTOP)
stopped=[]
try:
 deadline=time.monotonic()+5
 while state(coordinator) not in ['T','t']:
  assert time.monotonic()<deadline
  time.sleep(.02)
 q=json.loads((a/'queue_state.json').read_text())
 assert q['status']=='draining_predecessor_active_run' and q['runs']=={}
 old=json.loads((oldhere/'queue_state.json').read_text())
 assert old['current_run']=='075' and old['runs']['075']['status']=='running'
 child=old['runs']['075']['pid']
 attempt=next((Path(c['predecessor_root'])/'runs').glob('075-*'))/'artifacts/attempt-01-spark'
 assert str(attempt/'source/config.json') in command(child)
 policy.update(predecessor_state_before=old,queue_state_before=q,active_child_pid=child)
 (a/'finish_only_policy.json').write_text(json.dumps(policy,indent=2)+'\\n')
 for pid in [coordinator,admission]:
  os.kill(pid,signal.SIGTERM)
  os.kill(pid,signal.SIGCONT)
  deadline=time.monotonic()+5
  while state(pid) not in [None,'Z']:
   assert time.monotonic()<deadline, 'Admission scheduler did not exit'
   time.sleep(.05)
  stopped.append(pid)
 old.update(status='held_by_user_after_active_run',automatic_resume=False,user_hold_utc=policy['utc'])
 (oldhere/'queue_state.json').write_text(json.dumps(old,indent=2)+'\\n')
 env=dict(os.environ,PYTHONPATH=str(a.parents[1])+':'+c['remote_dependencies'])
 log=(a.parents[1]/'finish-only.log').open('x')
 p=subprocess.Popen([c['remote_python'],str(script)],cwd=a.parents[1],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt={'status':'finish_only_supervisor_launched','pid':p.pid,'stopped_admission_pids':stopped,'active_child_pid':child,'active_child_process_state':state(child),'resume_predecessor':False,'script_sha256':policy['replacement_script_sha256']}
 (a/'finish_only_launch.json').write_text(json.dumps(receipt,indent=2)+'\\n')
 print(json.dumps({'policy':policy,'launch':receipt}))
finally:
 if coordinator not in stopped and state(coordinator) in ['T','t']:
  (a/'finish_only_apply_failure.json').write_text(json.dumps({'status':'admission_held_for_recovery','automatic_resume':False}))
'''.replace('ANALYSIS', repr(remote)).replace('POLICY', repr(policy))
    result = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host,
        shlex.join([cohort['remote_python'], '-c', code])], text=True, timeout=45)
    data = json.loads(result)
    write_json(HERE / 'finish_only_policy.json', data['policy'])
    write_json(HERE / 'finish_only_launch.json', data['launch'])
    print(json.dumps(data['launch'], indent=2))


if __name__ == '__main__':
    main()
