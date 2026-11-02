"""Run215 capacity search: one scientific arm per independently qualified Pod."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cloud215', HERE/'03_cloud.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def separate_first_arm(state):
    pod = state['pods'][0]
    if pod.get('queued_arms_removed'):
        return
    output = c.outpath(pod)
    script = """python3 - <<'PY'
import json,os,signal,subprocess
from pathlib import Path
c=Path('/workspace/cdr215-control'); output=Path(OUTPUT)
assert not (output/'lr-0.001').exists() and not (output/'lr-0.003').exists()
workers=[]; drivers=[]; training=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try: args=(p/'cmdline').read_bytes().split(b'\\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if args[:2]==[b'bash',b'/workspace/cdr215-control/worker.sh']: workers.append(int(p.name))
 if args[:2]==[b'bash',b'/workspace/cdr215-control/driver.sh']: drivers.append(int(p.name))
 if args and args[0]==b'python3' and any(a.endswith(b'/01_train.py') for a in args):training.append(int(p.name))
assert len(workers)==len(drivers)==len(training)==1,(workers,drivers,training)
# Stop the waiting shell only. Its foreground Python/timeout child keeps running.
os.kill(workers[0],signal.SIGSTOP)
watch='''import json,os,signal,time
from pathlib import Path
p=Path(MANIFEST)
while True:
 try: m=json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError): m={}
 if m.get('status') in ['complete','failed','bounded_stop','loss_guard']:
  for pid in PIDS:
   try:os.kill(pid,signal.SIGKILL)
   except ProcessLookupError:pass
  Path('/workspace/cdr215-control/worker.exit').write_text('0' if m['status']=='complete' else '1')
  break
 time.sleep(10)
'''.replace('MANIFEST',repr(str(output/'lr-0.0003'/'manifest.json'))).replace('PIDS',repr(workers+drivers))
(c/'first-arm-watch.py').write_text(watch)
with (c/'first-arm-watch.log').open('w') as log:
 proc=subprocess.Popen(['python3',str(c/'first-arm-watch.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(stopped_waiting_shell=workers[0],driver=drivers[0],continuing_training=training[0],supervisor=proc.pid)
(c/'queue-separation.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
PY
""".replace('OUTPUT', repr(output))
    receipt = json.loads(c.ssh(pod, script))
    pod.update(rates=[0.0003], queued_arms_removed=True, queue_separation=receipt)
    c.write_json(c.SCRATCH/'production-lease.json', state)
    c.write_json(HERE/'production-parallelization.json', dict(utc=c.stamp(),
        reason='User prioritizes ongoing broad GPU search and parallel execution.',
        first_arm=receipt, hardware_policy='Mixed qualified GPUs; device recorded per arm.'))


def create(job, gpu, cloud, worker, state):
    task, rate = job
    price = gpu['securePricePerHr' if cloud=='SECURE' else 'communityPricePerHr']
    name = c.PREFIX+task+'-'+str(worker)
    public = (Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap = 'set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    epoch = time.time()
    actual = c.create_wait('pod','create','--name',name,'--image',state['image'],
        '--gpu-id',gpu['gpuId'],'--gpu-count','1','--cloud-type',cloud,
        '--min-cuda-version','13.0','--container-disk-in-gb','60','--volume-in-gb','20',
        '--ports','22/tcp','--ssh','--env',json.dumps(dict(PUBLIC_KEY=public)),
        '--docker-args','bash -lc '+shlex.quote(bootstrap),'--wait','--wait-timeout','20s')
    c.write_json(c.SCRATCH/('create-worker'+str(worker)+'.json'),actual)
    return dict(id=actual['id'],name=name,worker=worker,task=task,rates=[rate],gpu=gpu['displayName'],
        cloud_type=cloud,price_per_gpu_hour=price,created_epoch=epoch)


def main():
    state=c.lease()
    separate_first_arm(state)
    packets=json.loads((HERE/'production-packets.json').read_text())
    jobs=[('transformer',.001),('transformer',.003),('resnet',.0003),('resnet',.001),('resnet',.003)]
    # Never reuse worker numbers, including capacity failures without a Pod.
    previous=[int(p.stem.split('worker')[-1]) for p in c.SCRATCH.glob('capacity-worker*.json')]
    worker=max([p['worker'] for p in state['pods']]+previous)+1
    attempted={}; futures={}; last_search=0; last_status=0
    pool=ThreadPoolExecutor(max_workers=12)
    while True:
        for f,meta in list(futures.items()):
            if not f.done():continue
            del futures[f]
            kind=meta['kind']
            try:
                result=f.result()
                if kind=='create':
                    state['pods'].append(result)
                    print(json.dumps(dict(phase='created',worker=result['worker'],job=meta['job'],gpu=result['gpu'],cloud=result['cloud_type'])),flush=True)
                elif kind=='setup':
                    meta['pod']['launched_epoch']=time.time()
                    print(json.dumps(dict(phase='setup',worker=meta['pod']['worker'])),flush=True)
                elif kind=='retrieve':
                    print(json.dumps(dict(phase='retrieved',worker=meta['pod']['worker'])),flush=True)
            except Exception as error:
                if kind=='create':
                    c.write_json(c.SCRATCH/('capacity-worker'+str(meta['worker'])+'.json'),dict(utc=c.stamp(),job=meta['job'],gpu=meta['gpu'],cloud=meta['cloud'],error=repr(error)))
                    print(json.dumps(dict(phase='capacity_failed',worker=meta['worker'],gpu=meta['gpu'],cloud=meta['cloud'],error=str(error)[:180])),flush=True)
                elif kind=='setup':
                    pod=meta['pod'];pod['setup_error']=repr(error)
                    if not pod.get('delivery_started'):
                        actual=c.cli('pod','get',pod['id']);assert actual['name']==pod['name']
                        c.cli('pod','delete',pod['id']);pod['deleted_epoch']=time.time()
                        c.write_json(HERE/('production-retrieval-worker'+str(pod['worker'])+'.json'),dict(worker=pod['worker'],pod_id=pod['id'],verified=True,files=0,bytes=0,reason='No scientific inputs delivered; SSH startup failed.'))
                    elif not pod.get('driver_start_requested'):
                        c.ssh(pod,'echo 125 > '+c.CONTROL+'/worker.exit\n')
                        pod['driver_start_requested']=True
                    print(json.dumps(dict(phase='setup_failed',worker=pod['worker'],error=repr(error))),flush=True)
                else:meta['pod']['retrieval_error']=repr(error)
            c.write_json(c.SCRATCH/'production-lease.json',state)
        busy={m['pod']['worker'] for m in futures.values() if 'pod' in m}
        for pod in state['pods']:
            if not pod.get('deleted_epoch') and not pod.get('driver_start_requested') and pod['worker'] not in busy:
                f=pool.submit(c.setup_and_launch,pod,packets[pod['task']]);futures[f]=dict(kind='setup',pod=pod)
        if time.time()-last_status>30:
            for pod in state['pods']:
                if pod.get('deleted_epoch') or not pod.get('driver_start_requested') or pod['worker'] in busy:continue
                try:
                    current=c.status(pod)
                    c.write_json(c.SCRATCH/('status-worker'+str(pod['worker'])+'.json'),current)
                    pod['production_observed']=bool(current['arms'])
                    if current.get('worker_exit') is not None:
                        pod['worker_exit']=current['worker_exit']
                        f=pool.submit(c.retrieve_delete,pod);futures[f]=dict(kind='retrieve',pod=pod)
                except Exception as error:pod['last_monitor_error']=repr(error)
            last_status=time.time()
            c.write_json(c.SCRATCH/'production-lease.json',state)
        reserved={(p['task'],p['rates'][0]) for p in state['pods'] if not p.get('deleted_epoch') or p.get('worker_exit')=='0'}
        reserved.update(m['job'] for m in futures.values() if m['kind']=='create')
        pending=[j for j in jobs if j not in reserved]
        ready={(p['task'],p['rates'][0]) for p in state['pods'] if
            (p.get('production_observed') and not p.get('deleted_epoch') and p.get('worker_exit') is None)
            or p.get('worker_exit')=='0'}
        if all(j in ready for j in jobs) and not futures:break
        if not futures and (c.spend(state)>15 or time.time()>state['deadline_epoch']-14400):break
        if pending and time.time()-last_search>=25:
            try:catalog=c.cli('gpu','list','--include-unavailable')
            except (RuntimeError,subprocess.TimeoutExpired) as error:
                print(json.dumps(dict(phase='catalog_retry',error=repr(error))),flush=True)
                time.sleep(20);continue
            c.write_json(c.SCRATCH/'capacity-catalog-latest.json',dict(utc=c.stamp(),catalog=catalog))
            candidates=[]
            for gpu in catalog:
                if not gpu['available'] or gpu['memoryInGb']<8 or gpu['displayName'] in ['MI300X','Tesla V100','V100 SXM2','B200','B300']:continue
                for cloud,key in [('SECURE','securePricePerHr'),('COMMUNITY','communityPricePerHr')]:
                    price=gpu.get(key)
                    if price is not None and price<=5.29:candidates.append((gpu,cloud,price))
            for job in pending:
                if not candidates:break
                eligible=[r for r in candidates if not (job[0]=='transformer' and
                    r[0]['displayName']=='PRO 6000 MIG 24GB')]
                # This partition measured18s/update and failed the frozen4h gate.
                if not eligible:continue
                eligible.sort(key=lambda r:(attempted.get((r[0]['gpuId'],r[1]),0),r[0]['displayName']=='L4' and job[0]=='transformer',r[2]))
                gpu,cloud,price=eligible[0]
                attempted[(gpu['gpuId'],cloud)]=attempted.get((gpu['gpuId'],cloud),0)+1
                f=pool.submit(create,job,gpu,cloud,worker,state)
                futures[f]=dict(kind='create',job=job,gpu=gpu['displayName'],cloud=cloud,worker=worker);worker+=1
            last_search=time.time()
        time.sleep(3)
    pool.shutdown(wait=True)
    log=(c.SCRATCH/'monitor-parallel.log').open('a')
    proc=subprocess.Popen([sys.executable,str(HERE/'03_cloud.py'),'monitor'],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,creationflags=c.NO_WINDOW)
    c.write_json(c.SCRATCH/'parallel-controller-complete.json',dict(utc=c.stamp(),monitor_pid=proc.pid))


if __name__=='__main__':main()
