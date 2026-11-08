"""Replay verification of completed Spark branches; retrieve updated evidence."""
import importlib.util
from pathlib import Path
import json
import argparse

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('collection',HERE/'16_collect_spark.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);x=m.x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run');args=parser.parse_args()
    names=[args.run] if args.run else ['269-2026-11-07-mlp-representation-interventions','272-2026-11-07-mlp-displacement-confirmation']
    for run_name in names:
        run=x.ROOT/'runs'/run_name;relative=run.relative_to(x.ROOT).as_posix()
        result=x.remote(f'''import json,os,subprocess
from pathlib import Path
root=Path({x.REMOTE!r});worker={relative+'/artifacts/scientific/worker'!r}
assert json.loads((root/worker/'manifest.json').read_text())['status']=='completed'
env=dict(os.environ,PYTHONPATH={x.OLD['pythonpath']!r},CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with (root/{relative!r}/'scratch/representation-audit.log').open('w') as log:
 p=subprocess.run([{x.OLD['python']!r},{(HERE.relative_to(x.ROOT)/'18_verify_representations.py').as_posix()!r},'--run',{relative!r},'--worker',worker],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
assert p.returncode==0,(root/{relative!r}/'scratch/representation-audit.log').read_text()[-3000:]
r=json.loads((root/worker/'representation_audit.json').read_text());print(json.dumps(dict(status=r['status'],checks=len(r['checks']),max_absolute_error=r['max_absolute_error'])))''',timeout=660)
        x.dump(run/'representation_verification.json',result);print(run_name,result,flush=True)
        x.RUNS['mlp']=run;m.collect()


if __name__=='__main__':main()
