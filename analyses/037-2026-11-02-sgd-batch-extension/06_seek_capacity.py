"""Keep seeking task GPUs while the first workers calibrate; bounded by approval."""
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud037',HERE/'05_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PREFERENCE=['RTX 5090','RTX 4090','L40S','RTX PRO 5000','L40','A100 SXM','A100 PCIe','A40',
    'RTX PRO 4500','RTX PRO 4500 SE','PRO 6000 MIG 24GB','RTX 6000 Ada','RTX A6000']


def main():
    turn=0
    while not (c.SCRATCH/'STOP_CAPACITY').exists():
        state=c.lease();now=time.time()
        if state['start_epoch'] and now>=state.get('capacity_deadline_epoch',state['start_epoch']+1200):return
        active=[p for p in state['pods'] if not p.get('deleted_epoch')]
        if len(active)>=6:return
        slots=[(t,s) for s in range(3) for t in ('resnet','transformer') if not any(p['task']==t and p['seed']==s for p in active)]
        if not slots:return
        catalog=c.cli('gpu','list','--include-unavailable')
        choices=[]
        task,seed=slots[0]
        for name in PREFERENCE:
            if task=='transformer' and now>=state['start_epoch']+1800 and name!='RTX 5090':continue
            if (task=='transformer' and name=='PRO 6000 MIG 24GB') or (task=='resnet' and name in ('RTX PRO 4500','RTX PRO 4500 SE')):continue
            g=next((g for g in catalog if g['displayName']==name),None)
            if not g or g.get('memoryInGb',0)<16:continue
            for cloud,key in [('SECURE','securePricePerHr'),('COMMUNITY','communityPricePerHr')]:
                if task=='transformer' and seed==0 and cloud=='COMMUNITY':continue
                price=g.get(key)
                if price and price<=1.19:choices.append((not g['available'],PREFERENCE.index(name),name,cloud))
        choices.sort()
        available=[x for x in choices if not x[0]]
        attempts=available[:3] if available else [choices[turn%len(choices)]]
        for _,_,gpu,cloud in attempts:
            task,seed=slots[0]
            try:c.allocate(task,seed,gpu,cloud)
            except Exception as error:print(json.dumps(dict(utc=c.stamp(),error=repr(error))),flush=True)
            state=c.lease()
            if any(p['task']==task and p['seed']==seed and not p.get('deleted_epoch') for p in state['pods']):break
        turn+=1
        time.sleep(20)


if __name__=='__main__':main()
