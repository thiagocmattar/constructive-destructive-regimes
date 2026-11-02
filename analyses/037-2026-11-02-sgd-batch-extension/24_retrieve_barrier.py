"""Preserve the bounded inter-fork queue-wait receipt before worker cleanup."""
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
    while time.time()<c.lease()['scientific_deadline_epoch']:
        pod=next(p for p in c.lease()['pods'] if p['id']=='92xz66q1cvuby0')
        if pod.get('deleted_epoch'):raise RuntimeError('Worker deleted before barrier receipt retrieval')
        result=json.loads(c.ssh(pod,f'cat {c.CONTROL}/tail-barrier.json\n'))
        c.save(HERE/'tail-barrier.json',result)
        if 'finished_epoch' in result:
            print(json.dumps(result),flush=True);return
        time.sleep(10)
    raise RuntimeError('Barrier receipt retrieval reached its bound')


if __name__=='__main__':main()
