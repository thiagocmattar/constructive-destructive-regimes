"""Record owned-worker cleanup after accepted evidence and confirmed MCP deletion."""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]

def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
 row=next(r for r in load(HERE/'cohort.json') if r['run'][:3]==sys.argv[1])
 pre=ROOT/'runs'/row['run']/'prelaunch'; lease=load(pre/'lease.json')
 receipt=load(pre/(lease['attempt']+'-retrieval.json')); deleted=load(pre/'pod-delete.json')
 assert receipt['verified'] and receipt['scientific_verification']['temporal_verified']
 assert receipt['pod_id']==deleted['pod_id']==lease['pod_id']
 assert deleted['confirmed_absent']
 guard=load(pre/'deadline-process.json'); assert guard['pod_id']==lease['pod_id']
 command=f'''$taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId = {int(guard['pid'])}"
if ($null -eq $taskProcess) {{ 'GUARD_ALREADY_EXITED' }}
elseif ($taskProcess.CommandLine -like '*deadline-guard.ps1*' -and $taskProcess.CommandLine -like '*{lease['pod_id']}*') {{
 Stop-Process -Id {int(guard['pid'])} -Force; 'STOPPED_ATTRIBUTED_GUARD'
}} else {{ 'PID_REUSED_UNRELATED_PROCESS_PRESERVED' }}'''
 result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
 created=datetime.fromisoformat(lease['created_at'].replace('Z','+00:00'))
 end=datetime.fromisoformat(deleted['deleted_at'].replace('Z','+00:00'))
 hours=(end-created).total_seconds()/3600
 # Allocated storage: 60GB container + 40GB private pod volume; no network volume.
 gpu=hours*lease['price_per_hour']; storage=hours*100*.10/720
 record={**deleted,'created_at':lease['created_at'],'allocation_hours':hours,
  'price_per_hour':lease['price_per_hour'],'estimated_gpu_usd':gpu,
  'estimated_running_storage_usd':storage,'estimated_gpu_and_storage_usd':gpu+storage,
  'cost_provisional':True,'storage_month_hours_assumption':720,
  'storage_price_source':'https://docs.runpod.io/pods/storage/types',
  'artifact_acceptance':lease['attempt']+'-retrieval.json','retained_task_cloud_storage_gb':0,
  'deadline_guard_status':result.stdout.strip(),
  'retrieval_note':'Archive size/SHA, strict file inventory, frozen sources, complete coverage and independent CPU scientific checks passed before deletion.'}
 (pre/'closeout.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'run':row['run'][:3],'cost_usd_provisional':gpu+storage,
   'guard':result.stdout.strip(),'cloud_storage_gb':0}))

if __name__=='__main__':main()
