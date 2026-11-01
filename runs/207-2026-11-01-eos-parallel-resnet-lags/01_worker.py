"""Measure every assigned lag256 baseline first, then extend persistent profiles."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,stamp,tensor_hash,write_json
from src.eos_tasks import lag_profile
from src.eos_tasks.task import model_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--physical',type=int,default=512)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    path=Path(__file__).with_name('config.json');cfg=json.loads(path.read_text())
    jobs=[j for j in cfg['jobs'] if j['worker']==args.worker]
    manifest=dict(status='running',worker=args.worker,started_utc=stamp(),config_sha256=file_hash(path),windows=[],phase='common_coverage')
    write_json(args.output/'manifest.json',manifest)
    begin=time.perf_counter()

    def execute(phase,config):
        config_path=args.output/f'phase-config-{phase}.json';write_json(config_path,config)
        sys.argv=[str(Path(__file__)),'--worker',str(args.worker),'--output',str(args.output/phase),'--physical',str(args.physical)]
        lag_profile.main(config_path,progress_sink=args.output/'progress.json')

    try:
        base=dict(cfg,jobs=jobs,stages=[256])
        execute('baseline',base)
        shutil.copyfile(args.output/'baseline/environment.json',args.output/'environment.json')
        shutil.copyfile(args.output/'baseline/environment-lock.txt',args.output/'environment-lock.txt')
        baseline={}
        for job in jobs:
            key=f'{job["parent_run_id"]}-{job["anchor"]}';origin=args.output/'baseline'/key
            result=json.loads((origin/'result.json').read_text());baseline[key]=result
            shutil.copytree(origin,args.output/key)
            manifest['windows'].append(dict(parent_run_id=job['parent_run_id'],anchor=job['anchor'],last_lag=result['last_lag'],stop_reason=result['stop_reason']))
        write_json(args.output/'manifest.json',manifest)
        for job in jobs:
            key=f'{job["parent_run_id"]}-{job["anchor"]}';result=baseline[key]
            if result['last_lag']<256:continue
            if not job['initial_reversal'] or lag_profile.envelope(result['rows'],cfg['envelope_width'])<=cfg['damped_gap_threshold']:continue
            if (ROOT/cfg['stop_request']).exists():
                result['stop_reason']='budget_or_deadline_request';write_json(args.output/key/'result.json',result)
                next(w for w in manifest['windows'] if w['parent_run_id']==job['parent_run_id'] and w['anchor']==job['anchor'])['stop_reason']='budget_or_deadline_request'
                continue
            origin=args.output/'baseline'/key
            state=torch.load(origin/'end.pt',map_location='cpu',weights_only=False)
            model=model_for(job['task'],ROOT/cfg['task_cache']);model.load_state_dict(state['model'])
            extended=dict(job,resume_lag=256,checkpoint=(origin/'end.pt').relative_to(ROOT).as_posix(),
                checkpoint_sha256=file_hash(origin/'end.pt'),resume_tensor_sha256=tensor_hash(model.parameters()),
                boundary_gradient=(origin/'audit-current.npy').relative_to(ROOT).as_posix(),prefix_rows=result['rows'])
            del state,model
            manifest['phase']='extension';write_json(args.output/'manifest.json',manifest)
            phase='extension-'+key
            execute(phase,dict(cfg,jobs=[extended],stages=[512,1024,2048]))
            source=args.output/phase/key;later=json.loads((source/'result.json').read_text())
            if later['last_lag']>256:
                target=args.output/key
                for name in ['end.pt','audit-current.npy','audit-previous.npy']:
                    shutil.copyfile(source/name,target/name)
                if later['last_lag']==257:
                    shutil.copyfile(origin/'audit-current.npy',target/'audit-previous.npy')
                later['stages']=result['stages']+later['stages']
                later['resume_boundary_checks']=later['boundary_checks']
                later['boundary_checks']=result['boundary_checks']
                later['phase_sources']=dict(baseline=(origin/'result.json').relative_to(ROOT).as_posix(),
                    extension=(source/'result.json').relative_to(ROOT).as_posix(),
                    extension_checkpoint_sha256=extended['checkpoint_sha256'])
                later['elapsed_seconds']+=result['elapsed_seconds']
                write_json(target/'result.json',later)
                entry=next(w for w in manifest['windows'] if w['parent_run_id']==job['parent_run_id'] and w['anchor']==job['anchor'])
                entry.update(last_lag=later['last_lag'],stop_reason=later['stop_reason'])
                write_json(args.output/'manifest.json',manifest)
            else:
                result['stop_reason']='budget_or_deadline_request';write_json(args.output/key/'result.json',result)
                next(w for w in manifest['windows'] if w['parent_run_id']==job['parent_run_id'] and w['anchor']==job['anchor'])['stop_reason']='budget_or_deadline_request'
        manifest.update(status='complete',completed_utc=stamp(),elapsed_seconds=time.perf_counter()-begin,phase='complete')
        write_json(args.output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),completed_utc=stamp())
        write_json(args.output/'manifest.json',manifest)
        raise
    finally:
        write_json(args.output/'inventory.json',dict(files=[dict(path=p.relative_to(args.output).as_posix(),bytes=p.stat().st_size,sha256=file_hash(p))
            for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='inventory.json']))


if __name__=='__main__':main()
