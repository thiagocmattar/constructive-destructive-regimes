"""Prepare newly found task Pods without delaying already-running comparisons."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage216',HERE/'09_transformer512.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
c=t.c


def prepare(pod):
    if not pod.get('setup_started_epoch'):
        c.setup(pod)
    elif pod['task']=='transformer' and not pod.get('calibration216_started_epoch'):
        state=c.status(pod)
        if state.get('setup_exit')=='0':pod=t.stage(pod)
    elif pod['task']=='resnet' and not pod.get('calibration_started_epoch'):
        state=c.status(pod)
        if state.get('setup_exit')=='0':c.launch(pod,'calibration')
    return pod


def main():
    pending={};next_poll={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while time.time()<c.lease()['capacity_deadline_epoch']+300:
            state=c.lease()
            for pod in state['pods']:
                if pod.get('deleted_epoch') or pod.get('production_pid'):continue
                done=pod.get('calibration216_started_epoch') if pod['task']=='transformer' else pod.get('calibration_started_epoch')
                if done:continue
                if pod['id'] not in pending and time.time()>=next_poll.get(pod['id'],0):pending[pod['id']]=pool.submit(prepare,dict(pod))
            for key,future in list(pending.items()):
                if not future.done():continue
                try:
                    pod=future.result();latest=c.lease()
                    for item in latest['pods']:
                        if item['id']==pod['id']:item.update(pod)
                    c.save(c.SCRATCH/'lease.json',latest)
                    print(c.stamp(),pod['name'],'prepared',flush=True)
                except Exception as error:print(c.stamp(),key,repr(error),flush=True)
                del pending[key];next_poll[key]=time.time()+20
            time.sleep(3)


if __name__=='__main__':main()
