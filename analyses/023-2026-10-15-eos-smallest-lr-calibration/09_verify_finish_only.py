"""CPU-only success/failure checks for the user's no-follow-up-training policy."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json


def main():
    spec = importlib.util.spec_from_file_location('finish_only_test', HERE / '07_finish_only.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for failure in [False, True]:
        with tempfile.TemporaryDirectory(prefix='eos-finish-only-') as directory:
            root = Path(directory)
            new = root / 'priority'
            old = root / 'previous'
            here = new / 'analyses/023-fixture'
            oldhere = old / 'analyses/020-fixture'
            here.mkdir(parents=True)
            oldhere.mkdir(parents=True)
            attempt = old / 'runs/075-fixture/artifacts/attempt-01-spark'
            (attempt / 'production').mkdir(parents=True)
            write_json(attempt / 'production/manifest.json', dict(status='complete', completed_updates=500))
            module.ROOT, module.HERE = new, here
            records = [dict(run_id=x, kind='new_training') for x in ['120', '121']]
            cfg = dict(predecessor_root=str(old), predecessor_analysis='020-fixture',
                       drain_max_seconds=1, hard_queue_seconds=10, runs=records)
            previous = dict(status='running', current_run='075', runs={'075': dict(status='running', pid=987654321)})
            policy = dict(resume_predecessor=False, allowed_new_run_ids=['120', '121'],
                          predecessor_state_before=previous, utc=stamp(), instruction='Finish calibration, then wait for review.')
            calls = []
            def execute(*args):
                calls.append(args[0])
                assert args[0] == new and args[2] == records
                if failure: raise RuntimeError('CPU test failure')
            queue = SimpleNamespace(live=lambda pid: False, inventory=lambda attempt: None, execute=execute)
            try:
                module.finish_only(queue, cfg, policy, here / 'queue_state.json')
                assert not failure
            except RuntimeError:
                assert failure
            assert calls == [new], 'A predecessor admission path was invoked.'
            state = json.loads((oldhere / 'queue_state.json').read_text())
            assert state['automatic_resume'] is False and state['runs']['075']['status'] == 'complete'
            result = json.loads((here / 'finish_only_status.json').read_text())
            assert result['resume_predecessor'] is False
    write_json(HERE / 'finish_only_verification.json', dict(status='passed', utc=stamp(), cpu_only=True,
        optimizer_updates=0, script_sha256=file_hash(HERE / '07_finish_only.py'),
        checks=['Only the two authorized calibration IDs enter the training scheduler',
                'Active-run terminal evidence is retained',
                'Predecessor remains held after successful calibration',
                'Predecessor remains held on calibration failure']))
    print('PASS: finish-only success/failure paths keep all follow-up training held.')


if __name__ == '__main__':
    main()
