"""Read-only scientific/process/storage closeout of the owned Spark H049 task."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepare',HERE/'36_prepare_cap_lr_sweep.py')
x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)


def main():
    summary=json.loads((HERE/'cap_lr_summary.json').read_text());assert summary['status']=='passed'
    preserved=json.loads((HERE/'cap_lr_preserved_assets.json').read_text())
    for f in preserved['assets']:
        p=x.ROOT/f['path'];assert p.stat().st_size==f['bytes'] and x.sha(p)==f['sha256']
    compact=json.loads((HERE/'cap_lr_compact_retrieval.json').read_text());assert compact['status']=='passed'
    arm_seconds=[];audit_seconds=[];peak=0;local_bytes=0
    for run in x.RUNS:
        r=json.loads((run/'spark_compact_retrieval.json').read_text());assert r['status']=='passed'
        local_bytes+=sum(p.stat().st_size for p in (run/'artifacts').rglob('*') if p.is_file())
        if (run/'spark_retrieval.json').exists():
            historic=json.loads((run/'spark_retrieval.json').read_text())
            historic.update(full_local_copy_retained=False,current_retention='Raw binaries retained and reverified on Spark; local copies removed after the disk-limit recovery. See spark_compact_retrieval.json.')
            x.dump(run/'spark_retrieval.json',historic)
        for f in json.loads((run/'input_manifest.json').read_text())['files']:
            p=x.ROOT/f.get('local_path',f['path']);assert p.stat().st_size==f['bytes'] and x.sha(p)==f['sha256']
        w=run/'artifacts/attempt-01/worker';m=json.loads((w/'manifest.json').read_text())
        assert all(a['status']=='completed' for a in m['results'])
        for a in m['results']:arm_seconds.append(a['elapsed_seconds']);peak=max(peak,a['peak_cuda_bytes'])
        audit_seconds.append(json.loads((w/'cap_lr_replay.json').read_text())['elapsed_seconds'])
    pids=[json.loads((HERE/name).read_text())['pid'] for name in ['cap_lr_launch.json','cap_lr_audit_launch.json']]
    remote=x.remote(f'''from pathlib import Path
import subprocess,json,datetime
root=Path({x.REMOTE!r});scratch=root/'analyses/065-2026-11-07-mlp-frozen-lr-double/scratch'
training=json.loads((scratch/'h049-attempt-01-status.json').read_text());replay=json.loads((scratch/'h049-replay-status.json').read_text())
assert len(training['results'])==len(replay['results'])==3
assert all(r['returncode']==0 for r in training['results']+replay['results'])
processes={{}}
for pid in {pids!r}:
 p=subprocess.run(['ps','-p',str(pid),'-o','stat=,args='],capture_output=True,text=True)
 if p.returncode==0:assert p.stdout.strip().startswith('Z'),p.stdout
 processes[str(pid)]='exited' if p.returncode else 'exited (zombie awaiting parent reap)'
sizes=[]
for rel in {[r.relative_to(x.ROOT).as_posix() for r in x.RUNS]!r}:
 run=root/rel;sizes.append(dict(run=rel,bytes=sum(p.stat().st_size for p in (run/'artifacts').rglob('*') if p.is_file())))
print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=training,replay=replay,task_processes=processes,storage=sizes)))''')
    result=dict(status='passed',conditions=54,updates=54*705,training_driver_seconds=remote['training']['elapsed_seconds'],
        sum_arm_seconds=sum(arm_seconds),replay_seconds=sum(audit_seconds),peak_cuda_bytes=peak,
        local_scalar_history_bytes=local_bytes,raw_spark_bytes=sum(r['bytes'] for r in remote['storage']),
        exported_files=compact['files'],cloud_cost_usd=0,cloud_resources_created=0,
        retained_storage='Complete raw/recovery evidence on owned Spark. Verified compact outputs, scalar histories and replay records local; redundant transport archives removed.',
        storage_adjustment='Full local retrieval hit the disk limit; only verified remote duplicates were removed locally. Scientific inputs and outcomes unchanged.',
        scientific_inputs_unchanged=True,old_assets_unchanged=len(preserved['assets']),remote=remote)
    assert result['raw_spark_bytes']<12e9
    x.dump(HERE/'cap_lr_compute_closeout.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
