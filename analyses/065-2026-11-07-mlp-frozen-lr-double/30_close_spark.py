"""Verify task-owned Spark processes exited and retain their finished logs locally."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE/'14_prepare_transfer.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def main():
    jobs=[]
    for number in range(269,286):
        run=next((x.ROOT/'runs').glob(str(number)+'-*'))
        for path in run.glob('*_launch.json'):
            record=json.loads(path.read_text())
            if record.get('root')==x.REMOTE:
                jobs.append(dict(run=run.relative_to(x.ROOT).as_posix(),phase=record['phase'],pid=record['pid']))
    report=x.remote(f'''from pathlib import Path
import hashlib,json,tarfile,datetime
root=Path({x.REMOTE!r});jobs={jobs!r};checks=[];receipts=[]
for job in jobs:
 proc=Path('/proc')/str(job['pid'])/'cmdline';command=proc.read_bytes().replace(b'\\0',b' ').decode() if proc.exists() else ''
 assert job['run'] not in command, 'A task-owned Spark process is still active'
 manifest=root/job['run']/'artifacts'/job['phase']/'worker/manifest.json'
 checks.append(dict(**job,task_process_active=False,worker_status=json.loads(manifest.read_text()).get('status') if manifest.exists() else 'no worker manifest'))
for relative in sorted(set(job['run'] for job in jobs)):
 run=root/relative;files=[]
 for path in sorted((run/'scratch').glob('*.log')):
  with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=path.name,bytes=path.stat().st_size,sha256=digest))
 inventory=run/'scratch/spark-log-inventory.json';inventory.write_text(json.dumps(dict(files=files)))
 archive=run/'scratch/spark-logs.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for item in files:t.add(run/'scratch'/item['path'],arcname=item['path'])
  t.add(inventory,arcname='transfer_inventory.json')
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 receipts.append(dict(run=relative,archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(files),raw_bytes=sum(f['bytes'] for f in files)))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process_checks=checks,receipts=receipts)))''',timeout=180)
    for receipt in report['receipts']:
        run=x.ROOT/receipt['run'];target=run/'artifacts/spark-infrastructure';target.mkdir(parents=True,exist_ok=True)
        archive=target/'logs.tar.gz'
        subprocess.run(['scp','-q',x.HOST+':'+receipt['archive'],str(archive)],check=True,timeout=120)
        assert archive.stat().st_size==receipt['bytes'] and x.sha(archive)==receipt['sha256']
        with tarfile.open(archive) as t:t.extractall(target,filter='data')
        for item in json.loads((target/'transfer_inventory.json').read_text())['files']:
            path=target/item['path'];assert path.stat().st_size==item['bytes'] and x.sha(path)==item['sha256']
        assert archive.resolve().is_relative_to(target.resolve());archive.unlink()
        receipt.update(verified=True,local=target.relative_to(x.ROOT).as_posix(),duplicate_local_archive_removed=True)
        x.dump(run/'spark_logs_retrieval.json',receipt)
    report.update(status='verified',generator_sha256=x.sha(Path(__file__)),cloud_cost_usd=0,raw_evidence_retained_on='owned Spark')
    x.dump(HERE/'transfer_spark_closeout.json',report)
    print(json.dumps(dict(status='verified',processes=len(jobs),log_archives=len(report['receipts']))))


if __name__=='__main__':main()
