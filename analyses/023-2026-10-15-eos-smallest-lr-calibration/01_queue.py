"""Drain the active Spark run, run two priority pilots, then continue020 unchanged."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json
HERE = Path(__file__).resolve().parent


def verify_sources(root, analysis):
    for item in json.loads((root / 'analyses' / analysis / 'source_inventory.json').read_text())['files']:
        p = root / item['path']
        assert p.stat().st_size == item['bytes'] and file_hash(p) == item['sha256'], item['path']


def live(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def checked_command(pid, required):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert required in command, (pid, command, required)
    return command


def inventory(attempt):
    write_json(attempt / 'transfer_inventory.json', {'files': [
        dict(path=p.relative_to(attempt).as_posix(), bytes=p.stat().st_size, sha256=file_hash(p))
        for p in sorted(attempt.rglob('*')) if p.is_file() and p.name != 'transfer_inventory.json']})


def drain(cohort, statusfile):
    """Stop only admission; active child finishes with no pause in training time."""
    oldroot = Path(cohort['predecessor_root'])
    oldhere = oldroot / 'analyses' / cohort['predecessor_analysis']
    oldstate = oldhere / 'queue_state.json'
    original = json.loads(oldstate.read_text())
    if original['status'] in ['complete', 'finished_with_failures']:
        return original
    parent = cohort['predecessor_queue_pid']
    command = checked_command(parent, str(oldhere / '03_queue.py'))
    adopted = False
    os.kill(parent, signal.SIGSTOP)
    try:
        stop_deadline = time.monotonic() + 5
        while Path(f'/proc/{parent}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ['T', 't']:
            assert time.monotonic() < stop_deadline, 'Admission process did not stop.'
            time.sleep(.02)
        # Re-read after admission stops: the previous run may have just ended.
        state = json.loads(oldstate.read_text())
        write_json(HERE / 'handoff.json', dict(status='draining_active_run', utc=stamp(),
            parent_pid=parent, parent_command=command, predecessor_state_before=state,
            scientific_inputs_changed=False))
        started = time.monotonic()
        current = state.get('current_run')
        if current:
            record = state['runs'][current]
            if record['status'] == 'running':
                folder = next((oldroot / 'runs').glob(current + '-*'))
                attempt = folder / 'artifacts/attempt-01-spark'
                child = record['pid']
                if live(child):
                    checked_command(child, str(attempt / 'source/config.json'))
                while live(child):
                    if time.monotonic() - started > cohort['drain_max_seconds']:
                        raise TimeoutError('Active predecessor run did not drain; resume original queue.')
                    write_json(statusfile, dict(status='draining_predecessor_active_run', utc=stamp(),
                        predecessor_run=current, runs={}))
                    time.sleep(cohort.get('drain_poll_seconds', 30))
                manifest = json.loads((attempt / 'production/manifest.json').read_text())
                ok = manifest['status'] == 'complete' and manifest['completed_updates'] == 500
                record.update(status='complete' if ok else 'failed_before_priority_handoff',
                    ended_utc=manifest.get('completed_utc', stamp()))
                if ok: record['completed_updates'] = 500
                inventory(attempt)
        # Child is done. Replace only the scheduler; all scientific inputs stay frozen.
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        end = time.monotonic() + 15
        while live(parent) and time.monotonic() < end:
            time.sleep(.1)
        assert not live(parent), 'Old scheduler did not exit; no new GPU work admitted.'
        adopted = True
        state.update(status='held_for_priority_calibration', current_run=None,
                     infrastructure_handoff_utc=stamp())
        write_json(oldstate, state)
        write_json(HERE / 'handoff.json', dict(status='active_run_drained_scheduler_adopted', utc=stamp(),
            parent_pid=parent, predecessor_state_after=state, scientific_inputs_changed=False,
            continuation='Unstarted original020 runs resume after the priority calibration, with a fresh infrastructure deadline.'))
        return state
    finally:
        if not adopted and live(parent):
            os.kill(parent, signal.SIGCONT)


def execute(root, cohort, records, statusfile, state, hard_seconds):
    started = time.monotonic()
    state.update(status='running', resumed_or_started_utc=stamp())
    write_json(statusfile, state)
    for run in records:
        if run['kind'] == 'reused_gd_prefix':
            state['runs'].setdefault(run['run_id'], dict(status='reused_verified_source', updates=run['updates']))
            continue
        if run['run_id'] in state['runs']:
            continue  # An executed condition, including failure, is never repeated here.
        remaining = hard_seconds - (time.monotonic() - started)
        if remaining <= 0:
            state['runs'][run['run_id']] = dict(status='not_started_deadline')
            continue
        attempt = root / 'runs' / run['run'] / 'artifacts/attempt-01-spark'
        attempt.mkdir(parents=True, exist_ok=False)
        source = attempt / 'source'
        source.mkdir()
        for name in ['eos_tasks', 'cifar_eos']:
            shutil.copytree(root / 'src' / name, source / 'src' / name,
                            ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(root / 'runs' / run['run'] / 'config.json', source / 'config.json')
        assert file_hash(source / 'config.json') == run['config_sha256']
        command = [sys.executable, '-m', 'src.eos_tasks.train', '--config', str(source / 'config.json'),
                   '--cache', cohort['remote_cache'], '--mlp-cache', cohort['remote_mlp_cache'],
                   '--output', str(attempt / 'production')]
        env = dict(os.environ, PYTHONPATH=str(source) + ':' + cohort['remote_dependencies'])
        record = dict(status='running', started_utc=stamp(), task=run['task'], method=run['method'], seed=run['seed'])
        state['runs'][run['run_id']] = record
        state['current_run'] = run['run_id']
        with (attempt / 'production.log').open('x') as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            write_json(statusfile, state)
            try:
                code = child.wait(timeout=min(remaining, run['maximum_seconds']))
                assert code == 0, f'Trainer exit {code}'
                manifest = json.loads((attempt / 'production/manifest.json').read_text())
                assert manifest['status'] == 'complete' and manifest['completed_updates'] == run['updates']
                record.update(status='complete', completed_updates=run['updates'])
            except Exception as error:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                record.update(status='failed_or_deadline', error=repr(error))
            record['ended_utc'] = stamp()
        inventory(attempt)
        write_json(statusfile, state)
    complete = all(r['status'] in ['complete', 'reused_verified_source'] for r in state['runs'].values())
    state.update(status='complete' if complete else 'finished_with_failures', current_run=None,
                 completed_utc=stamp(), elapsed_this_scheduler_seconds=time.monotonic() - started)
    write_json(statusfile, state)


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    oldroot = Path(cohort['predecessor_root'])
    oldhere = oldroot / 'analyses' / cohort['predecessor_analysis']
    old = json.loads((oldhere / 'cohort.json').read_text())
    verify_sources(ROOT, HERE.name)
    verify_sources(oldroot, oldhere.name)
    statusfile = HERE / 'queue_state.json'
    assert not statusfile.exists(), 'Priority scheduler already started.'
    previous = drain(cohort, statusfile)
    try:
        execute(ROOT, cohort, cohort['runs'], statusfile,
                dict(status='starting', started_utc=stamp(), runs={}), cohort['hard_queue_seconds'])
    finally:
        # Fresh infrastructure deadline excludes the priority hold. No config or
        # scientific horizon changes, and no completed condition is re-executed.
        if previous['status'] not in ['complete', 'finished_with_failures']:
            write_json(HERE / 'predecessor_continuation.json', dict(status='resuming_unchanged_pending_runs', utc=stamp(),
                original_started_utc=previous['started_utc'], fresh_infrastructure_seconds=cohort['predecessor_continuation_seconds']))
            execute(oldroot, old, old['runs'], oldhere / 'queue_state.json', previous,
                    cohort['predecessor_continuation_seconds'])


if __name__ == '__main__':
    main()
