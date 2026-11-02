"""Place unchanged ResNet arms on newly available matched GPUs; preserve the active Transformer."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_cloud',HERE/'03_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    state=c.lease();assert not any(p['task']=='resnet' and not p.get('deleted_epoch') for p in state['pods'])
    catalog=c.cli('gpu','list','--include-unavailable')
    c.write_json(HERE/'scratch/resnet-capacity-refresh.json',catalog)
    public=(Path.home()/'.ssh/id_ed25519_runpod.pub').read_text().strip()
    bootstrap='set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server; mkdir -p /run/sshd /root/.ssh; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; sleep infinity'
    made=[]
    for name in ['RTX 5090','RTX PRO 4500','RTX 6000 Ada','RTX 4090','RTX PRO 5000','RTX PRO 4000']:
        gpu=next((g for g in catalog if g['displayName']==name and g['available'] and g.get('securePricePerHr') is not None and g['securePricePerHr']<=1.19),None)
        if not gpu:continue
        def create(index):
            started=time.time();podname=c.PREFIX+'resnet-'+str(index)
            actual=c.create_wait('pod','create','--name',podname,'--image',state['image'],'--gpu-id',gpu['gpuId'],
                '--gpu-count','1','--cloud-type','SECURE','--min-cuda-version','13.0',
                '--container-disk-in-gb','60','--volume-in-gb','20','--ports','22/tcp','--ssh',
                '--env',json.dumps(dict(PUBLIC_KEY=public)),'--docker-args','bash -lc '+shlex.quote(bootstrap),
                '--wait','--wait-timeout','20s')
            return dict(id=actual['id'],name=podname,worker=index+1,task='resnet',gpu=name,
                price_per_gpu_hour=gpu['securePricePerHr'],created_epoch=started)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(create,i):i for i in range(3)}
            for future in as_completed(futures):
                try:
                    pod=future.result();made.append(pod);state['pods'].append(pod)
                    c.write_json(HERE/'scratch/production-lease.json',state)
                    print(json.dumps(dict(phase='resnet_created',worker=pod['worker'],gpu=name)),flush=True)
                except Exception as error:
                    c.write_json(HERE/('scratch/resnet-create-'+name.replace(' ','')+'-'+str(futures[future])+'.json'),dict(error=repr(error)))
        if made:break
    made.sort(key=lambda p:p['worker'])
    if made:
        rates=json.loads((HERE/'config.json').read_text())['learning_rates_by_task']['resnet']
        for pod in made:pod['rates']=[]
        for i,rate in enumerate(rates):made[i%len(made)]['rates'].append(rate)
        c.write_json(HERE/'scratch/production-lease.json',state)
        packet=json.loads((HERE/'production-packets.json').read_text())['resnet']
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(c.setup_and_launch,pod,packet):pod for pod in made}
            for future in as_completed(futures):
                pod=futures[future]
                try:future.result();pod['launched_epoch']=time.time()
                except Exception as error:pod['setup_error']=repr(error)
                c.write_json(HERE/'scratch/production-lease.json',state)
                print(json.dumps(dict(phase='resnet_setup',worker=pod['worker'],error=pod.get('setup_error'))),flush=True)
    else:
        c.write_json(HERE/'resnet-placement-pending.json',dict(status='no_capacity',utc=c.stamp()))
    log=(HERE/'scratch/cloud-controller-monitor.log').open('w')
    subprocess.Popen([sys.executable,'-u',str(HERE/'03_cloud.py'),'monitor'],stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=c.NO_WINDOW)
    log.close()


if __name__=='__main__':main()
