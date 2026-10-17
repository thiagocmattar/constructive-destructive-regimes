"""Stop each task-owned GPU after every run assigned to it is locally verified."""
from pathlib import Path
import importlib.util
import json
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('h021_control', HERE / '04_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def assigned_runs(families):
    return [name for family in families for name in [family['baseline'], *family['branches']]]


def all_verified(root, names):
    if not names:
        return False
    for name in names:
        try:
            receipt = json.loads((root / 'runs' / name / 'prelaunch/attempt-01-retrieval.json').read_text())
        except (OSError, ValueError):
            return False
        if not receipt.get('verified') or not receipt.get('scientific_verification', {}).get('verified'):
            return False
    return True


def budget_snapshot(lease, maximum_usd, observed_at):
    hourly = sum(p['hourly_price'] for p in lease['pods']) + lease['storage_hourly_price']
    estimated_cost = max(0, observed_at - lease['created_at_epoch']) / 3600 * hourly
    return dict(estimated_task_cost_usd=estimated_cost,
                estimated_budget_remaining_usd=max(0, maximum_usd-estimated_cost),
                assumed_hourly_usd=hourly,
                automatic_cutoff_epoch=min(lease['hard_deadline_epoch'],
                    lease['created_at_epoch'] + maximum_usd / hourly * 3600),
                assumption='Both GPUs active since first allocation, plus estimated storage; provisional billing')


def stop_verified(pod, names):
    if not all_verified(ROOT, names):
        return None
    before = control.assert_owned(pod)
    if before.get('runtimeStatus') not in {'stopped', 'terminated'}:
        control.cli('pod', 'stop', pod['id'])
    after = control.assert_owned(pod)
    return dict(pod=pod['id'], worker=pod['worker'], runs=names,
                requested_at_epoch=time.time(), runtimeStatus=after.get('runtimeStatus'),
                desiredStatus=after.get('desiredStatus'))


def main():
    lease = json.loads((HERE / 'lease.json').read_text())
    assert lease['owner'] == 'Analysis025-current-task'
    authorization = json.loads((HERE / 'authorization.json').read_text())
    assert authorization['approved']
    plan = json.loads((HERE / 'worker_plan.json').read_text())
    names = {p['id']: assigned_runs(plan[p['worker']]) for p in lease['pods']}
    records = {}
    while time.time() < lease['hard_deadline_epoch']:
        if (HERE / 'closeout.json').exists():
            return
        for pod in lease['pods']:
            previous = records.get(pod['id'], {})
            if previous.get('runtimeStatus') in {'stopped', 'terminated'}:
                continue
            try:
                result = stop_verified(pod, names[pod['id']])
                if result:
                    records[pod['id']] = result
                    print(json.dumps(result), flush=True)
            except Exception as error:
                records[pod['id']] = dict(pod=pod['id'], at_epoch=time.time(), error=repr(error))
                print(json.dumps(records[pod['id']]), flush=True)
        observed_at = time.time()
        control.save(HERE / 'idle-stop-status.json', dict(observed_at_epoch=observed_at,
            budget=budget_snapshot(lease, authorization['maximum_total_usd'], observed_at), pods=records))
        time.sleep(30)


if __name__ == '__main__':
    main()
