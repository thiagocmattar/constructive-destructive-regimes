"""Reduce beside verified Spark checkpoints; retrieve compact reproducible outputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepare',HERE/'36_prepare_cap_lr_sweep.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def main():
    scripts=[HERE/'17_reduce_transfer.py',HERE/'39_reduce_cap_lr.py']
    rel=HERE.relative_to(x.ROOT).as_posix()
    # These are post-run readers; no executed scientific input is changed.
    for p in scripts:subprocess.run(x.SCP+[str(p),x.HOST+':'+x.REMOTE+'/'+p.relative_to(x.ROOT).as_posix()],check=True,timeout=180)
    result=x.remote(f'''from pathlib import Path
import subprocess,os,json,hashlib,time
root=Path({x.REMOTE!r});analysis=root/{rel!r}
for name,digest in {[(p.name,x.sha(p)) for p in scripts]!r}:
 assert hashlib.sha256((analysis/name).read_bytes()).hexdigest()==digest
env=dict(os.environ,PYTHONPATH={x.OLD['pythonpath']!r},OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
started=time.monotonic()
p=subprocess.run([{x.OLD['python']!r},str(analysis/'39_reduce_cap_lr.py')],cwd=root,env=env,capture_output=True,text=True,timeout=600)
(analysis/'scratch/h049-reduction.log').write_text(p.stdout+p.stderr)
assert p.returncode==0,p.stdout+p.stderr
print(json.dumps(dict(status='passed',seconds=time.monotonic()-started,output=p.stdout)))''',timeout=650)
    x.dump(HERE/'cap_lr_remote_reduction.json',result);print(json.dumps(result),flush=True)
    receipt=x.remote(f'''from pathlib import Path
import hashlib,json,tarfile
root=Path({x.REMOTE!r});analysis=root/{rel!r};exports=[];retained=[]
def record(p):
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest)
for run_rel in {[r.relative_to(x.ROOT).as_posix() for r in x.RUNS]!r}:
 run=root/run_rel;artifact=run/'artifacts';logs=artifact/'execution_logs';logs.mkdir(exist_ok=True)
 for name in ['attempt-01.log','mechanics-01.log','cap-lr-replay.log']:
  source=run/'scratch'/name
  if source.exists():(logs/name).write_bytes(source.read_bytes())
 files=[record(p) for p in sorted(artifact.rglob('*')) if p.is_file() and p.name!='transfer_inventory.json']
 inventory=run/'spark_retained_inventory.json'
 inventory.write_text(json.dumps(dict(root=str(root),status='verified',files=files)))
 retained.append(dict(run=run_rel,files=len(files),bytes=sum(f['bytes'] for f in files)))
 exports += [record(inventory),record(run/'reduction_verification.json')]
 exports += [f for f in files if Path(f['path']).suffix in ['.json','.jsonl','.log'] and Path(f['path']).name!='updates.jsonl']
for name in ['cap_lr_curves.npz','cap_lr_summary.json','cap_lr_per_seed.csv','cap_lr_aggregate.csv','cap_lr_statistics.md','cap_lr_table.tex']:
 exports.append(record(analysis/name))
manifest=analysis/'cap_lr_export_manifest.json';manifest.write_text(json.dumps(dict(files=exports,retained=retained,scope='All raw evidence on owned Spark; local reduced curves/statistics, boundary histories, evaluations, metadata and replay records. Per-unit update streams and binary checkpoints remain on Spark.')))
archive=analysis/'scratch/h049-compact-results.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for f in exports:tar.add(root/f['path'],arcname=f['path'])
 tar.add(manifest,arcname=manifest.relative_to(root).as_posix())
receipt=record(archive);receipt['remote_archive']=str(archive);receipt['files']=len(exports);receipt['retained']=retained
print(json.dumps(receipt))''',timeout=600)
    archive=HERE/'scratch/h049-compact-results.tar.gz'
    subprocess.run(x.SCP+[x.HOST+':'+receipt['remote_archive'],str(archive)],check=True,timeout=600)
    assert archive.stat().st_size==receipt['bytes'] and x.sha(archive)==receipt['sha256']
    with tarfile.open(archive) as tar:tar.extractall(x.ROOT,filter='data')
    for f in json.loads((HERE/'cap_lr_export_manifest.json').read_text())['files']:
        p=x.ROOT/f['path'];assert p.stat().st_size==f['bytes'] and x.sha(p)==f['sha256'],str(p)
    receipt.update(status='passed',archive_verified=True,all_exported_file_sizes_and_sha256_verified=True,
        raw_retention='Complete verified raw evidence on owned Spark; compact outputs and scalar histories local.',
        local_storage_adjustment='cap_lr_local_storage_adjustment.json',remote_reduction=result)
    x.dump(HERE/'cap_lr_compact_retrieval.json',receipt)
    for run in x.RUNS:
        x.dump(run/'spark_compact_retrieval.json',dict(status='passed',receipt=(HERE/'cap_lr_compact_retrieval.json').relative_to(x.ROOT).as_posix(),raw_inventory='spark_retained_inventory.json',raw_binaries_on_spark=True))
    cleanup=x.remote(f'''from pathlib import Path
import json
root=Path({x.REMOTE!r});removed=[]
for rel in {[r.relative_to(x.ROOT).as_posix()+'/scratch/h049-results.tar' for r in x.RUNS]+[rel+'/scratch/h049-compact-results.tar.gz']!r}:
 p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and p.name in ['h049-results.tar','h049-compact-results.tar.gz']
 if p.exists():p.unlink();removed.append(rel)
print(json.dumps(dict(removed_redundant_remote_transport_archives=removed)))''')
    archive.unlink();receipt.update(cleanup);receipt['local_transport_archive_removed']=True
    x.dump(HERE/'cap_lr_compact_retrieval.json',receipt)
    print(json.dumps(dict(status='passed',exported_files=receipt['files'],archive_bytes=receipt['bytes'])),flush=True)


if __name__=='__main__':main()
