"""Prefetch sealed early audit snapshots; final retrieval still verifies every file."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import threading
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'15_cloud_transfer.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
RECEIPT=HERE/'early_audit_transfers.json';LOCK=threading.Lock()


def record(item):
    with LOCK:
        data=json.loads(RECEIPT.read_text()) if RECEIPT.exists() else dict(status='running',generator_sha256=c.x.file_hash(Path(__file__)),transfers=[])
        key=(item['name'],item['local'])
        if key not in [(r['name'],r['local']) for r in data['transfers']]:data['transfers'].append(item)
        c.dump(RECEIPT,data)
    print(json.dumps(item),flush=True)


def worker(name):
    while True:
        pod=c.get(name)
        if pod.get('deleted_epoch'):return
        try:
            with LOCK:
                known=[r['local'] for r in json.loads(RECEIPT.read_text())['transfers'] if r['name']==name] if RECEIPT.exists() else []
            jobs=[j for j in pod['jobs'] if not j['mechanics'] and not j.get('verified_retrieval')]
            code=f'''python3 - <<'PY'
from pathlib import Path
import json,hashlib
root=Path({c.REMOTE!r});result=[];known={known!r}
for job in {jobs!r}:
 cfg=json.loads((root/job['run_dir']/'config.json').read_text());out=root/job['output']
 for arm in job['arms']:
  folder=out/arm;manifest=folder/'manifest.json';progress=folder/'progress.json'
  if not manifest.exists() or not progress.exists():continue
  try:status=json.loads(manifest.read_text());p=json.loads(progress.read_text())
  except json.JSONDecodeError:continue
  if status['status']!='running' or p['state']>cfg['end']-64:continue
  for state in [cfg['fork'],cfg['fork']+32]:
   path=folder/('audit_'+str(state)+'.pt');relative=path.relative_to(root).as_posix()
   if relative in known or p['state']<=state or not path.exists():continue
   with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   result.append(dict(remote=str(path),local=relative,bytes=path.stat().st_size,sha256=digest,arm=arm,audit_state=state,observed_completed_state=p['state'],label=job['label']))
print(json.dumps(result))
PY
'''
            candidates=json.loads(c.x.ssh(pod,code,timeout=180))
            for item in candidates:
                target=c.ROOT/item['local'];target.parent.mkdir(parents=True,exist_ok=True)
                allowed=[c.ROOT/'runs',Path('R:/cdr-h048-evidence/runs')]
                assert any(target.resolve().is_relative_to(p.resolve()) for p in allowed)
                partial=target.with_name(target.name+'.prefetch.partial');begin=time.monotonic();downloaded=False
                if target.exists():
                    assert target.stat().st_size==item['bytes'] and c.x.file_hash(target)==item['sha256']
                else:
                    c.x.copy(pod,partial,item['remote'],False)
                    assert partial.stat().st_size==item['bytes'] and c.x.file_hash(partial)==item['sha256']
                    if target.exists():assert target.stat().st_size==item['bytes'] and c.x.file_hash(target)==item['sha256']
                    else:os.replace(partial,target)
                    downloaded=True
                if partial.exists():
                    assert partial.parent.resolve()==target.parent.resolve();partial.unlink()
                record(dict(item,name=name,verified=True,downloaded=downloaded,elapsed_seconds=time.monotonic()-begin,utc=c.x.stamp()))
        except Exception as error:
            if c.get(name).get('deleted_epoch'):return
            print(json.dumps(dict(name=name,event='prefetch_retry',error=str(error)[-1000:])),flush=True)
        time.sleep(30)


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,['transformer-a','transformer-b','transformer-c2']))
    with LOCK:
        data=json.loads(RECEIPT.read_text());data.update(status='completed',finished_utc=c.x.stamp());c.dump(RECEIPT,data)


if __name__=='__main__':main()
