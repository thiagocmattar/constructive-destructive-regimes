"""Sequential detached workers with at most two independent retrievals in flight."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def transport(run):
    spec=importlib.util.spec_from_file_location('transport_'+run.name[:3],run/'02_remote.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);return r


def status(r):
    return r.remote(f'''import json
from pathlib import Path
out=Path({r.REMOTE!r})/{r.REL!r}/'artifacts/attempt-01/worker'
record=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else dict(status='starting')
record['progress']=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else None
print(json.dumps(record))''')


def main():
    plan=json.loads((HERE/'selective_plan.json').read_text());futures={};records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for item in plan['arms']:
            assert not (HERE/'STOP_SELECTIVE').exists()
            run=ROOT/item['path'];r=transport(run)
            if item['target_run'] in futures:futures[item['target_run']].result()
            if not (run/'verification_prelaunch.json').exists():r.stage()
            if not (run/'spark_launch.json').exists():r.launch()
            latest=None
            while True:
                assert not (HERE/'STOP_SELECTIVE').exists()
                q=status(r)
                progress=q.get('progress') or {}
                if progress.get('state')!=latest:
                    print(json.dumps(dict(arm=item['arm'],status=q['status'],progress=progress)),flush=True);latest=progress.get('state')
                if q['status'] in ['completed','failed']:break
                time.sleep(2)
            assert q['status']=='completed',q
            records.append(dict(arm=item['arm'],path=item['path'],manifest=q))
            if not (run/'retrieval.json').exists():futures[item['path']]=pool.submit(r.collect)
            else:assert json.loads((run/'retrieval.json').read_text())['verified']
            print(json.dumps(dict(completed=item['arm'],seconds=q['elapsed_seconds'])),flush=True)
        for future in futures.values():future.result()
    result=dict(status='completed',runs=records,all_retrievals_verified=True)
    (HERE/'selective_execution.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(dict(status='completed',workers=len(records),total_worker_seconds=sum(q['manifest']['elapsed_seconds'] for q in records))),flush=True)


if __name__=='__main__':main()
