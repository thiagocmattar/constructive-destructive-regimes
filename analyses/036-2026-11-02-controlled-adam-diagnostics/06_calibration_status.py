"""Bounded read-only status for the Run213 calibration lease."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calibration_cloud', HERE/'05_cloud_calibration.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)

state = cloud.lease()
for pod in state['pods']:
    row = dict(worker=pod['worker'], task=pod['task'], gpu=pod['gpu'], rates=pod.get('rates'))
    if pod.get('deleted_epoch'):
        row['deleted'] = True
    else:
        try:
            current = cloud.cli('pod', 'get', pod['id'])
            row['runtime_status'] = current.get('runtimeStatus')
            route = current.get('ssh', {})
            if route.get('ip'):
                pod.update(host=route['ip'], port=route['port'])
                details = cloud.status(pod)
                if details.get('setup_exit') == '0':
                    details.pop('setup_tail', None)
                if details.get('arms'):
                    details.pop('worker_tail', None)
                row.update(details)
        except Exception as error:
            row['error'] = repr(error)
    print(json.dumps(row), flush=True)
print(json.dumps(dict(estimated_compute_storage_usd=cloud.spend(state))), flush=True)
