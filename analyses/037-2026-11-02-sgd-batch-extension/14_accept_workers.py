"""Accept complete local workers independently, then release only their task Pods."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


c=module('cloud037','05_cloud.py')
v=module('verify037','02_verify_results.py')


def accept(pod,cleanup=False):
    receipt_path=HERE/('stream-retrieval-'+pod['name']+'.json')
    if not receipt_path.exists():return False
    receipt=c.read(receipt_path)
    if receipt['scientific_status']!='complete':return False
    target=c.ROOT/receipt['path']
    assert receipt['pod_id']==pod['id'] and receipt['verified']
    assert c.file_hash(target/'worker_inventory.json')==receipt['inventory_sha256']
    results=v.verify_worker(target)
    assert len(results)==(50 if pod['task']=='resnet' else 66)
    acceptance=dict(status='passed',utc=c.stamp(),pod_id=pod['id'],task=pod['task'],seed=pod['seed'],
        inventory_sha256=receipt['inventory_sha256'],verified_forks=len(results),
        verifier_sha256=c.file_hash(HERE/'02_verify_results.py'),
        config_sha256=c.file_hash(c.runpath(pod)/'config.json'),
        input_manifest_sha256=c.file_hash(c.runpath(pod)/'input_manifest.json'))
    c.save(HERE/('worker-acceptance-'+pod['name']+'.json'),acceptance)
    c.save(HERE/('fork-reduction-'+pod['name']+'.json'),dict(status='passed',forks=results))
    if cleanup:
        c.delete(pod)
        latest=c.lease();next(p for p in latest['pods'] if p['id']==pod['id']).update(pod)
        c.save(c.SCRATCH/'lease.json',latest)
    print(json.dumps(dict(**acceptance,deleted=bool(pod.get('deleted_epoch')))),flush=True)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cleanup',action='store_true')
    p.add_argument('--watch',action='store_true')
    args=p.parse_args();accepted=set()
    while True:
        state=c.lease()
        for pod in state['pods']:
            if pod.get('deleted_epoch') or not pod.get('production_pid') or pod['id'] in accepted:continue
            try:
                if accept(dict(pod),args.cleanup):accepted.add(pod['id'])
            except Exception as error:
                print(c.stamp(),pod['name'],repr(error),flush=True)
        if not args.watch or time.time()>=state['deadline_epoch'] or (c.SCRATCH/'closed.json').exists():break
        time.sleep(30)


if __name__=='__main__':main()
