"""Wait for the task-owned queue, replay diagnostics, retrieve, then delete pod."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud',HERE/'15_cloud_transfer.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
spec=importlib.util.spec_from_file_location('stream',HERE/'23_stream_retrieval.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)


def main():
    a=argparse.ArgumentParser();a.add_argument('--name',required=True);args=a.parse_args();pod=c.get(args.name)
    while True:
        with contextlib.redirect_stdout(io.StringIO()):state=c.status(args.name)
        progress=[dict(label=j['label'],exit=j['exit'],progress=j.get('progress')) for j in state['jobs']]
        print(json.dumps(dict(utc=c.x.stamp(),name=args.name,jobs=progress)),flush=True)
        for job in state['jobs']:
            if job.get('arms'):stream.collect(args.name,job['label'],'shared')
            spec=next(j for j in c.get(args.name)['jobs'] if j['label']==job['label'])
            requires_reuse=bool(json.loads((c.ROOT/spec.get('run_dir',c.RUNS[spec['case']])/'config.json').read_text()).get('baseline_reference'))
            for name,status in job.get('arms',{}).items():
                if name=='control' and requires_reuse and not job.get('control_reuse_ready'):continue
                if status['status']=='completed':stream.collect(args.name,job['label'],name)
        if all(j['exit'] is not None for j in state['jobs']):break
        time.sleep(30)
    observer=HERE/'18_verify_representations.py';rel=observer.relative_to(c.ROOT).as_posix()
    c.x.ssh(pod,'mkdir -p '+c.REMOTE+'/'+observer.parent.relative_to(c.ROOT).as_posix()+'\n')
    c.x.copy(pod,observer,c.REMOTE+'/'+rel)
    for job in c.get(args.name)['jobs']:
        if job.get('verified_retrieval'):continue
        status=next(j for j in state['jobs'] if j['label']==job['label'])
        if status['exit']=='0' and not job['mechanics']:
            command=f'''cd {c.REMOTE}
export PYTHONPATH={c.REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/opt/cdr-transfer/bin/python {rel} --run {job.get('run_dir',c.RUNS[job['case']])} --worker {job['output']} > /workspace/{job['label']}-audit.log 2>&1
'''
            c.x.ssh(pod,command,timeout=900)
            print(json.dumps(dict(name=args.name,label=job['label'],representation_audit='passed')),flush=True)
            run=job.get('run_dir',c.RUNS[job['case']]);cfg=json.loads((c.ROOT/run/'config.json').read_text())
            if any(a['kind']=='budget' for a in cfg['arms']):
                budget_observer=c.ROOT/'runs/278-2026-11-07-mlp-functional-step-budget/02_audit_budget.py';budget_rel=budget_observer.relative_to(c.ROOT).as_posix()
                c.x.ssh(pod,'mkdir -p '+c.REMOTE+'/'+budget_observer.parent.relative_to(c.ROOT).as_posix()+'\n')
                c.x.copy(pod,budget_observer,c.REMOTE+'/'+budget_rel)
                command=f'''cd {c.REMOTE}
export PYTHONPATH={c.REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/opt/cdr-transfer/bin/python {budget_rel} --run {run} --worker {job['output']} >> /workspace/{job['label']}-audit.log 2>&1
'''
                c.x.ssh(pod,command,timeout=900)
                print(json.dumps(dict(name=args.name,label=job['label'],functional_budget_audit='passed')),flush=True)
            if any(a['kind']=='loss_guard' for a in cfg['arms']):
                guard_observer=HERE/'28_audit_probe_loss.py';guard_rel=guard_observer.relative_to(c.ROOT).as_posix()
                c.x.copy(pod,guard_observer,c.REMOTE+'/'+guard_rel)
                command=f'''cd {c.REMOTE}
export PYTHONPATH={c.REMOTE} CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/opt/cdr-transfer/bin/python {guard_rel} --run {run} --worker {job['output']} >> /workspace/{job['label']}-audit.log 2>&1
'''
                c.x.ssh(pod,command,timeout=900)
                print(json.dumps(dict(name=args.name,label=job['label'],probe_loss_audit='passed')),flush=True)
        stream.collect(args.name,job['label'])
    c.delete(args.name)
    print(json.dumps(dict(name=args.name,status='closed',exits={j['label']:j['exit'] for j in state['jobs']})),flush=True)


if __name__=='__main__':main()
