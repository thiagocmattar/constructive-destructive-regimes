"""Read terminal scalar results ahead of the independently verified full transfer."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import time

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('cloud034',HERE/'01_cloud.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def one(worker):
    while True:
        pod=next(p for p in reversed(c.lease()['pods']) if p['worker']==worker and not p.get('deleted_epoch'))
        if (HERE/f'calibration-scalar-worker{worker}.json').exists() or c.receipt_path(pod,'209').exists():return
        state=c.status(pod,'209')
        if state.get('worker_exit')=='0':c.capture_calibration(pod);return
        time.sleep(20)


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(one,[0,3,4,5,6,7,8]))
