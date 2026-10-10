"""Record cleanup only after accepted evidence and attributed MCP deletion."""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs'/sys.argv[1]
assert run.name in [r['run'] for r in json.loads((ROOT/'cache/runpod/seed-cohort.json').read_text())]
p=run/'prelaunch'
receipt=json.loads((p/'attempt1-retrieval.json').read_text())
deleted=json.loads((p/'pod-delete.json').read_text())
lease=json.loads((p/'lease.json').read_text())
assert receipt['verified'] and receipt['scientific_verification']['verified']
assert receipt['pod_id']==deleted['pod_id']==lease['pod_id']
assert deleted['confirmed_absent'] and deleted['deletion_status']==204
guard=json.loads((p/'deadline-process.json').read_text())
assert guard['pod_id']==lease['pod_id']
pid=int(guard['pid']); pod=lease['pod_id']; assert pod.isalnum()
command=f'''$taskGuard=Get-CimInstance Win32_Process -Filter "ProcessId = {pid}";
if ($null -eq $taskGuard) {{ 'ALREADY_ABSENT' }}
elseif ($taskGuard.CommandLine.Contains('{pod}') -and $taskGuard.CommandLine.Contains('{run.name}')) {{
 Stop-Process -Id {pid} -Force; 'STOPPED_ATTRIBUTED_GUARD'
}} else {{ 'PID_REUSED_UNRELATED_PROCESS_PRESERVED' }}'''
r=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
guard_state=r.stdout.strip()
created=datetime.fromisoformat(lease['created_at'].replace('Z','+00:00'))
end=datetime.fromisoformat(deleted['deleted_at'].replace('Z','+00:00'))
hours=(end-created).total_seconds()/3600
gpu=hours*lease['price_per_hour']; storage=hours*(lease['container_gb']+lease['persistent_gb'])*.10/720
record={**deleted,'created_at':lease['created_at'],'allocation_hours':hours,
    'price_per_hour':lease['price_per_hour'],'estimated_gpu_usd':gpu,
    'estimated_running_storage_usd':storage,'estimated_gpu_and_storage_usd':gpu+storage,
    'cost_provisional':True,'storage_price_source':'https://docs.runpod.io/pods/storage/types',
    'artifact_acceptance':'attempt1-retrieval.json','retained_task_cloud_storage_gb':0,
    'deadline_guard_status':guard_state,
    'retrieval_note':'Exact archive bytes/SHA, strict file inventory, independent CPU scientific checks, and initial tensor hash accepted before deletion.'}
(p/'closeout.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'run':run.name[:3],'cost_usd_provisional':gpu+storage,'guard':guard_state,'cloud_storage_gb':0}))
