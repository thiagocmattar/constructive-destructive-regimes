"""Read-only Spark closeout for the eleven selective-update study runs."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    plan = json.loads((HERE / 'selective_plan.json').read_text())
    execution = json.loads((HERE / 'selective_execution.json').read_text())
    assert execution['status'] == 'completed' and execution['all_retrievals_verified']
    paths = [plan['calibration_run']] + [item['path'] for item in plan['arms']]
    spec = importlib.util.spec_from_file_location('selective_transport', ROOT / paths[0] / '02_remote.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    remote = transport.remote(f'''
import datetime,json,subprocess
from pathlib import Path
root=Path({transport.REMOTE!r}); paths={paths!r}
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
workers=[line for line in processes if '/01_worker.py' in line and any(p in line for p in paths)]
runs=[]
for rel in paths:
    run=root/rel
    manifest=json.loads((run/'artifacts/attempt-01/worker/manifest.json').read_text())
    runs.append(dict(path=rel,bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),manifest=manifest))
source_archives=[root/('run'+Path(p).name[:3]+'-source.tar.gz') for p in paths]
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=workers,runs=runs,
    root_source_archive_bytes=sum(p.stat().st_size for p in source_archives if p.exists()))))
''')
    assert not remote['workers'], remote['workers']
    assert all(item['manifest']['status'] == 'completed' for item in remote['runs'])
    receipts = [json.loads((ROOT / p / 'retrieval.json').read_text()) for p in paths]
    assert all(r['verified'] and r['status'] == 'completed' for r in receipts)
    remote['status'] = 'exited'
    remote['retrieved_files'] = sum(r['files'] for r in receipts)
    remote['retrieved_payload_bytes'] = sum(r['file_bytes'] for r in receipts)
    remote['retrieved_archive_bytes'] = sum(r['bytes'] for r in receipts)
    remote['retained_spark_bytes'] = sum(r['bytes'] for r in remote['runs']) + remote['root_source_archive_bytes']
    remote['retained_local_run_bytes'] = sum(p.stat().st_size for rel in paths for p in (ROOT / rel).rglob('*') if p.is_file())
    remote['total_worker_seconds'] = sum(r['manifest']['elapsed_seconds'] for r in remote['runs'])
    remote['peak_allocated_bytes'] = max(r['manifest']['peak_allocated_bytes'] for r in remote['runs'])
    remote['cloud_cost_usd'] = 0
    remote['billing_note'] = 'Owned Spark only; no cloud resources provisioned.'
    remote['cleanup'] = 'All task workers exited. Raw evidence, recovery states and archives retained on Spark and locally; unrelated resources unchanged.'
    remote['storage_budget_bytes_per_machine'] = 30_000_000_000
    assert remote['retained_spark_bytes'] < remote['storage_budget_bytes_per_machine']
    assert remote['retained_local_run_bytes'] < remote['storage_budget_bytes_per_machine']
    (HERE / 'selective_compute_closeout.json').write_text(json.dumps(remote, indent=2) + '\n', newline='\n')
    print(json.dumps({k: v for k, v in remote.items() if k not in ['runs']}, indent=2))


if __name__ == '__main__':
    main()
