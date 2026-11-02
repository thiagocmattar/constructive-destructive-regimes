"""Recover two never-started arms onto warm matching 5090 workers, unchanged inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cloud_relocation', HERE/'05_cloud_calibration.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def launch(pod, task, rate):
    while True:
        result = cloud.status(pod)
        if result.get('worker_exit') is not None:
            assert result['worker_exit'] == '0', result
            break
        time.sleep(10)
    if task == 'transformer':
        archive = json.loads((HERE/'calibration-packets.json').read_text())[task]
        cloud.copy(pod, cloud.ROOT/archive['path'], '/workspace/cdr213-transformer-inputs.tar.gz')
        cloud.ssh(pod, "echo '"+archive['sha256']+"  /workspace/cdr213-transformer-inputs.tar.gz' | sha256sum -c -\ntar -xzf /workspace/cdr213-transformer-inputs.tar.gz -C "+cloud.REMOTE+'\n')
    output = cloud.outpath(pod)+'/lr-'+str(rate)+('-transformer' if task == 'transformer' else '')
    relative = cloud.RUN.relative_to(cloud.ROOT).as_posix()
    body = f'''set -uo pipefail
cd {cloud.REMOTE}
export PATH=/opt/cdr213-venv/bin:$PATH OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
timeout --signal=TERM 3600 python3 {relative}/01_train.py --task {task} --lr {rate} --output {output} > {cloud.CONTROL}/reused-worker.log 2>&1
echo $? > {cloud.CONTROL}/worker.exit
'''
    local = cloud.SCRATCH/('reuse-worker'+str(pod['worker'])+'.sh')
    local.write_text(body, encoding='utf-8', newline='\n')
    cloud.copy(pod, local, cloud.CONTROL+'/reuse.sh')
    cloud.ssh(pod, f'cp {cloud.CONTROL}/worker.exit {cloud.CONTROL}/original-worker.exit\nrm {cloud.CONTROL}/worker.exit\nsetsid bash {cloud.CONTROL}/reuse.sh > {cloud.CONTROL}/reuse-driver.log 2>&1 < /dev/null &\necho $!\n')
    return dict(task=task, lr=rate, output=output, started_epoch=time.time())


def main():
    state = cloud.lease()
    for worker in (1, 2):
        pod = next(p for p in state['pods'] if p['worker'] == worker)
        actual = cloud.cli('pod', 'get', pod['id'])
        assert actual['name'] == pod['name'] and actual['name'].startswith(cloud.PREFIX)
        assert not pod.get('launched_epoch') and not pod.get('host')
        assert actual.get('runtimeStatus') == 'initializing', actual.get('runtimeStatus')
        read = subprocess.run(['runpodctl', 'pod', 'logs', pod['id'], '--tail', '30', '--max-wait', '2s'],
            capture_output=True, text=True, timeout=20, creationflags=cloud.NO_WINDOW)
        log = cloud.SCRATCH/('never-started-worker'+str(worker)+'.txt')
        log.write_text(read.stdout+read.stderr, encoding='utf-8')
        cloud.write_json(HERE/('calibration-retrieval-worker'+str(worker)+'.json'),
            dict(worker=worker, pod_id=pod['id'], verified=True, files=0, bytes=0,
                reason='Container never reported; no SSH or scientific input/code delivery occurred.',
                infrastructure_log_sha256=cloud.file_hash(log)))
        cloud.cli('pod', 'delete', pod['id'])
        pod['deleted_epoch'] = time.time()
        pod['stop_reason'] = 'never_started_relocated_unchanged_arm'
    cloud.write_json(cloud.SCRATCH/'calibration-lease.json', state)
    jobs = [(next(p for p in state['pods'] if p['worker'] == 3), 'transformer', .001),
        (next(p for p in state['pods'] if p['worker'] == 4), 'resnet', .0003)]
    cloud.write_json(HERE/'calibration-relocation-plan.json', dict(
        reason='Two container starts never became ready; use warm same-class workers after their ResNet arms.',
        scientific_inputs_changed=False, run_id='213', jobs=[dict(worker=p['worker'], task=t, lr=r) for p,t,r in jobs]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, pod, task, rate): pod for pod,task,rate in jobs}
        for future in as_completed(futures):
            pod = futures[future]
            pod['extra_job'] = future.result()
            cloud.write_json(cloud.SCRATCH/'calibration-lease.json', state)
            print(json.dumps(dict(phase='relocated', worker=pod['worker'], job=pod['extra_job'])), flush=True)
    output = (cloud.SCRATCH/'cloud-controller-monitor.log').open('w')
    subprocess.Popen([sys.executable, '-u', str(HERE/'05_cloud_calibration.py'), 'monitor'],
        stdout=output, stderr=subprocess.STDOUT, creationflags=cloud.NO_WINDOW)
    output.close()


if __name__ == '__main__':
    main()
