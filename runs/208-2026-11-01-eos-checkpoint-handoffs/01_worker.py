"""Resume a verified measured prefix on an independent qualified worker."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,write_json
from src.eos_tasks import lag_profile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--physical',type=int,default=512)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    path=Path(__file__).with_name('config.json');cfg=json.loads(path.read_text())
    job=next(j for j in cfg['jobs'] if j['worker']==args.worker)
    key=f'{job["parent_run_id"]}-{job["anchor"]}'
    manifest=dict(status='running',worker=args.worker,started_utc=stamp(),config_sha256=file_hash(path),windows=[])
    write_json(args.output/'manifest.json',manifest);begin=time.perf_counter()
    try:
        sys.argv=[str(Path(__file__)),'--worker',str(args.worker),'--output',str(args.output/'continuation'),'--physical',str(args.physical)]
        lag_profile.main(path,progress_sink=args.output/'progress.json')
        child=args.output/'continuation';source=child/key;target=args.output/key
        shutil.copytree(source,target)
        for name in ['environment.json','environment-lock.txt']:shutil.copyfile(child/name,args.output/name)
        prefix_path=ROOT/job['prefix_result'];prefix=json.loads(prefix_path.read_text())
        result=json.loads((source/'result.json').read_text())
        assert result['rows'][:job['resume_lag']+1]==prefix['rows']
        shutil.copyfile(prefix_path.with_name('audit-first.npy'),target/'audit-first.npy')
        if result['last_lag']==job['resume_lag']:
            for name in ['audit-previous.npy','audit-current.npy']:shutil.copyfile(prefix_path.with_name(name),target/name)
        elif result['last_lag']==job['resume_lag']+1:
            shutil.copyfile(prefix_path.with_name('audit-current.npy'),target/'audit-previous.npy')
        result['stages']=prefix['stages']+[s for s in result['stages'] if s['lag'] not in {v['lag'] for v in prefix['stages']}]
        result['resume_boundary_checks']=result['boundary_checks'];result['boundary_checks']=prefix['boundary_checks']
        result['handoff_sources']=dict(prefix=job['prefix_result'],resume_lag=job['resume_lag'],
            checkpoint_sha256=job['checkpoint_sha256'],prefix_result_sha256=file_hash(prefix_path))
        result['elapsed_seconds']+=prefix['elapsed_seconds']
        write_json(target/'result.json',result)
        manifest.update(status='complete',completed_utc=stamp(),elapsed_seconds=time.perf_counter()-begin,
            windows=[dict(parent_run_id=job['parent_run_id'],anchor=job['anchor'],last_lag=result['last_lag'],stop_reason=result['stop_reason'])])
        write_json(args.output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),completed_utc=stamp());write_json(args.output/'manifest.json',manifest)
        raise
    finally:
        write_json(args.output/'inventory.json',dict(files=[dict(path=p.relative_to(args.output).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p))
            for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
