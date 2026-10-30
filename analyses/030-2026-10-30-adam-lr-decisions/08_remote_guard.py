"""Pod-local hard deadline; uses only the container's own Runpod credential."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def own_identity(expected, environ_bytes):
    values = dict(item.split('=', 1) for item in environ_bytes.decode().split('\0') if '=' in item)
    assert values.get('RUNPOD_POD_ID') == expected, 'Foreign pod rejected'
    assert values.get('RUNPOD_API_KEY'), 'Missing pod-local credential'
    return values['RUNPOD_API_KEY']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert 0 < args.deadline - time.time() <= 12 * 3600
    key = own_identity(args.pod, Path('/proc/1/environ').read_bytes())
    url = 'https://api.runpod.io/v2/pods/' + args.pod
    log = Path('/workspace/constructive/worker-control/remote-guard.jsonl')

    def record(event, **fields):
        with log.open('a') as stream:
            stream.write(json.dumps(dict(event=event, epoch=time.time(), **fields)) + '\n')

    def request(action=None):
        headers = {'Authorization': 'Bearer ' + key}
        payload = None
        if action is not None:
            headers['Content-Type'] = 'application/json'
            payload = json.dumps({'action': action}).encode()
        req = urllib.request.Request(url + ('/action' if action else ''), data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read()
            return json.loads(body) if body else {}

    pod = request()
    assert pod['id'] == args.pod and 'stop' in pod['actions']
    record('armed', pod=args.pod, deadline=args.deadline, readback_verified=True)
    while time.time() < args.deadline:
        time.sleep(max(.01, min(10, args.deadline - time.time())))
    while True:
        try:
            record('stop_requested', pod=args.pod)
            request('stop')
            record('stop_accepted', pod=args.pod)
            return
        except Exception as exc:
            # Never log request headers, credentials, or provider response bodies.
            record('stop_retry', error_type=type(exc).__name__)
            time.sleep(10)


if __name__ == '__main__':
    main()
