"""Acquire the five planned Transformer followers near prefix readiness."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import os
import time

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fleet',HERE/'58_matched_fleet.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)


def main():
    os.environ['H051_TRANSFER_VIA_SPARK']='1'
    while True:
        p=f.get('transformer-1');assert not p.get('deleted_epoch') and not p.get('budget_stopped')
        state=f.status('transformer-1')
        if state.get('prefix_exit') not in [None,'0']:raise RuntimeError(state)
        if state.get('prefix_exit')=='0' or state.get('progress',{}).get('state',0)>=300:break
        time.sleep(30)
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs=[]
        for number in range(2,7):
            name='transformer-'+str(number)
            if not (f.S/('lease-'+name+'.json')).exists():
                for attempt in range(12):
                    try:f.create(name,'transformer');break
                    except Exception as error:
                        f.emit(event='allocation_retry',name=name,attempt=attempt+1,error=str(error)[-400:])
                        if attempt==11:raise
                        time.sleep(20)
            p=f.get(name)
            if p['phase']=='created':jobs.append(pool.submit(f.setup,name))
        for job in jobs:job.result()
    f.emit(event='planned_transformer_fleet_ready',workers=6)


if __name__=='__main__':main()
