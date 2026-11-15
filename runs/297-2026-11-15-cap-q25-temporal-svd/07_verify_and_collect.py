"""Bounded read-only completion monitor, independent verification and retrieval."""
import importlib.util,json,shlex,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('transport',HERE/'03_spark.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
analysis=ROOT/'analyses/065-2026-11-07-mlp-frozen-lr-double'
for case in ['resnet','transformer']:
    deadline=time.monotonic()+14400
    while True:
        assert time.monotonic()<deadline,'Completion monitor deadline reached'
        command='cd '+shlex.quote(t.REMOTE)+'\n'
        command+='if test -f '+t.REL+'/scratch/production-'+case+'.exit; then echo COMPLETE; else '
        command+='if test -f '+t.REL+'/artifacts/attempt-01/'+case+'/progress.json; then cat '+t.REL+'/artifacts/attempt-01/'+case+'/progress.json; else echo WAITING; fi; fi\n'
        result=t.ssh(command).strip()
        if result=='COMPLETE':break
        print(case,result,flush=True)
        time.sleep(60)
    command='cd '+shlex.quote(t.REMOTE)+'\nexport OPENBLAS_NUM_THREADS=4\n'
    command+=' '.join(map(shlex.quote,[t.L['python'],t.REL+'/06_verify_dense.py','--case-root',t.REL+'/artifacts/attempt-01/'+case]))+'\n'
    print(t.ssh(command),flush=True)
    subprocess.run([sys.executable,str(analysis/'70_collect_q25_svd.py'),case],check=True)
    subprocess.run([sys.executable,str(analysis/'71_report_q25_svd.py')],check=True)
print('All remaining cases verified, retrieved and plotted.',flush=True)
