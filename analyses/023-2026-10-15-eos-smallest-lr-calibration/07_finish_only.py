"""User-directed scheduler override: finish075/120/121, then leave training held."""
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import stamp, write_json


def finish_only(queue, cohort, policy, statusfile):
    assert policy['resume_predecessor'] is False
    assert policy['allowed_new_run_ids'] == ['120', '121']
    assert [r['run_id'] for r in cohort['runs'] if r['kind'] == 'new_training'] == ['120', '121']
    oldroot = Path(cohort['predecessor_root'])
    oldstatefile = oldroot / 'analyses' / cohort['predecessor_analysis'] / 'queue_state.json'
    previous = policy['predecessor_state_before']
    current = previous.get('current_run')
    if current:
        record = previous['runs'][current]
        assert current == '075' and record['status'] == 'running'
        attempt = next((oldroot / 'runs').glob(current + '-*')) / 'artifacts/attempt-01-spark'
        child = record['pid']
        if queue.live(child):
            queue.checked_command(child, str(attempt / 'source/config.json'))
        begin = time.monotonic()
        while queue.live(child):
            assert time.monotonic()-begin < cohort['drain_max_seconds'], 'Active run drain deadline; pending jobs remain held.'
            write_json(statusfile, dict(status='draining_predecessor_active_run', utc=stamp(),
                predecessor_run=current, runs={}, resume_predecessor=False))
            time.sleep(30)
        manifest = json.loads((attempt / 'production/manifest.json').read_text())
        ok = manifest['status'] == 'complete' and manifest.get('completed_updates') == 500
        record.update(status='complete' if ok else 'failed_before_calibration',
            ended_utc=manifest.get('completed_utc', stamp()))
        if ok: record['completed_updates'] = 500
        queue.inventory(attempt)
    previous.update(status='held_by_user_pending_calibration_review', current_run=None,
        user_hold_utc=policy['utc'], automatic_resume=False)
    write_json(oldstatefile, previous)
    write_json(HERE / 'predecessor_continuation.json', dict(status='disabled_by_user', utc=stamp(),
        resume_predecessor=False, reason=policy['instruction']))
    try:
        queue.execute(ROOT, cohort, cohort['runs'], statusfile,
            dict(status='starting', started_utc=stamp(), runs={}, resume_predecessor=False),
            cohort['hard_queue_seconds'])
    finally:
        # There is deliberately no admission path for any predecessor condition.
        write_json(HERE / 'finish_only_status.json', dict(status='no_further_training_authorized', utc=stamp(),
            allowed_new_run_ids=['120', '121'], resume_predecessor=False,
            next_action='Retrieve and analyze completed calibration; await user review.'))


def main():
    spec = importlib.util.spec_from_file_location('qualified_calibration_queue', HERE / '01_queue.py')
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    cohort = json.loads((HERE / 'cohort.json').read_text())
    policy = json.loads((HERE / 'finish_only_policy.json').read_text())
    queue.verify_sources(ROOT, HERE.name)
    queue.verify_sources(Path(cohort['predecessor_root']), cohort['predecessor_analysis'])
    finish_only(queue, cohort, policy, HERE / 'queue_state.json')


if __name__ == '__main__':
    main()
