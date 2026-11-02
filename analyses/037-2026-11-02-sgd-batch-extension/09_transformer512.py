"""Stage Run216's frozen microbatch config and qualify each short cloud queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
RUN216=c.ROOT/'runs/216-2026-11-02-sgd-transformer-microbatch512'
REL216=RUN216.relative_to(c.ROOT).as_posix()


def stage(pod):
    c.connect(pod)
    archive=c.SCRATCH/'run216-code.tar.gz'
    c.copy(pod,archive,'/workspace/run216-code.tar.gz')
    c.ssh(pod,f"echo '{c.file_hash(archive)}  /workspace/run216-code.tar.gz' | sha256sum -c -\ntar -xf /workspace/run216-code.tar.gz -C {c.REMOTE}\n")
    body=f'''set -euo pipefail
trap 'echo $? > {c.CONTROL}/calibration216.exit' EXIT
cd {c.REMOTE}
export PATH=/opt/cdr214-venv/bin:$PATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python {REL216}/00_profile.py --task transformer --full --output {c.CONTROL}/profile216.json
python {REL216}/02_worker.py --task transformer --seed {pod['seed']} --smoke --output {c.CONTROL}/short-worker216 --deadline-epoch {time.time()+600}
'''
    script=c.SCRATCH/(pod['name']+'-cal216.sh');script.write_text(body,encoding='utf-8',newline='\n')
    c.copy(pod,script,c.CONTROL+'/calibration216.sh')
    pid=c.ssh(pod,f'set -euo pipefail\ntest ! -e {c.CONTROL}/calibration216.pid\nsetsid bash {c.CONTROL}/calibration216.sh > {c.CONTROL}/calibration216.log 2>&1 < /dev/null &\necho $! | tee {c.CONTROL}/calibration216.pid\n').strip()
    pod.update(run_id='216',calibration216_started_epoch=time.time(),calibration216_pid=pid)
    return pod


def verify(pod):
    c.connect(pod)
    c.copy(pod,HERE/'02_verify_results.py',c.CONTROL+'/verify.py')
    script=f'''test "$(cat {c.CONTROL}/calibration216.exit)" = 0
/opt/cdr214-venv/bin/python - <<'PY'
import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,{c.REMOTE!r})
s=importlib.util.spec_from_file_location('verification',{(c.CONTROL+'/verify.py')!r})
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
v.ROOT=Path('/workspace');v.RUN=Path({(c.REMOTE+'/'+REL216)!r})
r=v.verify_worker(Path({(c.CONTROL+'/short-worker216')!r}),mechanics=True,input_manifest=v.RUN/'input_manifest.json')
print(json.dumps(dict(status='passed',mechanics_only=True,verified_forks=len(r))))
PY
'''
    result=json.loads(c.ssh(pod,script,timeout=120))
    result.update(utc=c.stamp(),pod_id=pod['id'],seed=pod['seed'],run_id='216',
        config_sha256=c.file_hash(RUN216/'config.json'),input_manifest_sha256=c.file_hash(RUN216/'input_manifest.json'))
    c.save(HERE/f'cloud-qualification216-seed{pod["seed"]}.json',result)
    print(json.dumps(result),flush=True)
    return pod


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','verify']);p.add_argument('--seeds',type=int,nargs='+',required=True)
    a=p.parse_args();pods=[next(p for p in reversed(c.lease()['pods']) if p['task']=='transformer' and p['seed']==seed and not p.get('deleted_epoch')) for seed in a.seeds]
    with ThreadPoolExecutor(max_workers=3) as pool:updated=list(pool.map(stage if a.action=='stage' else verify,pods))
    latest=c.lease()
    for pod in updated:
        for current in latest['pods']:
            if current['id']==pod['id']:current.update(pod)
    c.save(c.SCRATCH/'lease.json',latest)


if __name__=='__main__':main()
