import json,os,signal,time
from pathlib import Path
p=Path('/workspace/constructive/runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01/worker0/lr-0.0003/manifest.json')
while True:
 try: m=json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError): m={}
 if m.get('status') in ['complete','failed','bounded_stop','loss_guard']:
  for pid in [1667, 1559]:
   try:os.kill(pid,signal.SIGKILL)
   except ProcessLookupError:pass
  Path('/workspace/cdr215-control/worker.exit').write_text('0' if m['status']=='complete' else '1')
  break
 time.sleep(10)
