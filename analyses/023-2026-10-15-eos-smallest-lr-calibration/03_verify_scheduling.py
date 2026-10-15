"""Linux CPU-only test of admission drain, no lost child, and unchanged continuation."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    spec = importlib.util.spec_from_file_location('priority_queue_test', HERE / '01_queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    with tempfile.TemporaryDirectory(prefix='eos-priority-cpu-') as temporary:
        root = Path(temporary)
        old = root / 'old'
        oldhere = old / 'analyses/020-fixture'
        own = root / 'priority'
        oldhere.mkdir(parents=True)
        own.mkdir()
        queue.HERE = own
        attempt = old / 'runs/075-fixture/artifacts/attempt-01-spark'
        (attempt / 'production').mkdir(parents=True)
        (attempt / 'source').mkdir()
        (attempt / 'source/config.json').write_text('{}')
        worker = old / 'worker.py'
        worker.write_text('import time,json,sys; from pathlib import Path; time.sleep(4); '
            'Path(sys.argv[2]).write_text(json.dumps({"status":"complete","completed_updates":500}))')
        parentfile = oldhere / '03_queue.py'
        parentfile.write_text('import json,subprocess,sys; from pathlib import Path; '
            f'p=subprocess.Popen([sys.executable,{str(worker)!r},{str(attempt / "source/config.json")!r},'
            f'{str(attempt / "production/manifest.json")!r}]); '
            f'Path({str(oldhere / "queue_state.json")!r}).write_text(json.dumps({{"status":"running",'
            '"started_utc":"fixture","current_run":"075","runs":{"075":{"status":"running","pid":p.pid}}})); '
            f'p.wait(); Path({str(old / "incorrect-next-admission")!r}).write_text("bad")')
        parent = subprocess.Popen([sys.executable, str(parentfile)], stdout=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while not (oldhere / 'queue_state.json').exists():
                assert time.monotonic() < deadline
                time.sleep(.02)
            state = queue.drain(dict(predecessor_root=str(old), predecessor_analysis=oldhere.name,
                predecessor_queue_pid=parent.pid, drain_max_seconds=10, drain_poll_seconds=.05), own / 'queue_state.json')
            parent.wait(timeout=5)
            assert state['runs']['075']['status'] == 'complete'
            assert state['status'] == 'held_for_priority_calibration'
            assert not (old / 'incorrect-next-admission').exists()
            assert (attempt / 'transfer_inventory.json').exists()
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait()
        # Execute real CPU subprocesses through the production scheduler interface.
        taskroot = root / 'tasks'
        (taskroot / 'src/eos_tasks').mkdir(parents=True)
        (taskroot / 'src/cifar_eos').mkdir()
        (taskroot / 'src/eos_tasks/train.py').write_text(
            'import argparse,json; from pathlib import Path; p=argparse.ArgumentParser(); '
            '[p.add_argument("--"+k) for k in ["config","cache","mlp-cache","output"]]; '
            'a=p.parse_args(); c=json.loads(Path(a.config).read_text()); o=Path(a.output); '
            'o.mkdir(); (o/"manifest.json").write_text(json.dumps({"status":"complete","completed_updates":c["updates"]}))')
        records = []
        for rid in ['120', '121']:
            folder = taskroot / 'runs' / rid
            folder.mkdir(parents=True)
            write_json(folder / 'config.json', dict(updates=3))
            records.append(dict(run_id=rid, run=rid, kind='new_training', task='cpu_fixture',
                method='gd', seed=0, updates=3, maximum_seconds=20,
                config_sha256=file_hash(folder / 'config.json')))
        cfg = dict(remote_cache='', remote_mlp_cache='', remote_dependencies='')
        output = own / 'fixture_state.json'
        state = dict(runs={})
        queue.execute(taskroot, cfg, records, output, state, 60)
        assert state['status'] == 'complete'
        # Completed conditions must be skipped, despite existing attempt dirs.
        queue.execute(taskroot, cfg, records, output, state, 60)
        assert all(r['status'] == 'complete' for r in state['runs'].values())
        assert list(state['runs']) == ['120', '121']
    write_json(HERE / 'scheduling_verification.json', dict(status='passed', utc=stamp(),
        cpu_only=True, optimizer_updates=0, queue_script_sha256=file_hash(HERE / '01_queue.py'),
        checks=['Active child finishes while only admissions are stopped',
                'No next predecessor run starts during priority handoff',
                'Completed child and transfer inventory retained',
                'Priority jobs execute sequentially with persisted terminal manifests',
                'Continuation skips already executed conditions']))
    print('PASS: CPU-only drain, exclusive admission, persisted child, sequential priority, no replay')


if __name__ == '__main__':
    main()
