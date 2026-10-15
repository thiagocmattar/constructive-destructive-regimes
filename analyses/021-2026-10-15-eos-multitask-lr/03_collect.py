"""Reuse the hash-checked retriever, selecting each evidence-owning remote root."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_local(run):
    folder = ROOT / 'runs' / run
    receiptfile = folder / 'retrieval_verification.json'
    if not receiptfile.exists():
        return False
    receipt = json.loads(receiptfile.read_text())
    attempt = folder / 'artifacts/attempt-01-spark'
    if receipt['status'] != 'passed' or file_hash(attempt / 'transfer_inventory.json') != receipt['inventory_sha256']:
        return False
    for item in json.loads((attempt / 'transfer_inventory.json').read_text())['files']:
        p = attempt / item['path']
        assert p.resolve().is_relative_to(attempt.resolve())
        if not p.exists() or p.stat().st_size != item['bytes'] or file_hash(p) != item['sha256']:
            return False
    return True


def verify_legacy(record):
    analysis = ROOT / 'analyses/019-2026-10-13-cifar10-eos-lr'
    receipt = json.loads((analysis / 'retrieval.json').read_text())
    assert receipt['status'] == 'complete' and record['run_id'] in receipt['runs']
    attempt = ROOT / 'runs' / record['run'] / 'artifacts/attempt-01-spark'
    inventory = json.loads((attempt / 'transfer_inventory.json').read_text())
    for item in inventory['files']:
        p = attempt / item['path']
        assert p.resolve().is_relative_to(attempt.resolve())
        assert p.stat().st_size == item['bytes'] and file_hash(p) == item['sha256']
    return dict(run=record['run'], status='passed', files=len(inventory['files']),
        inventory_sha256=file_hash(attempt / 'transfer_inventory.json'), source_analysis='019')


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    previous = ROOT / 'analyses/020-2026-10-14-eos-multitask'
    tool = load_module(previous / '05_retrieve.py', 'eos_retrieve')
    old_remote = tool.REMOTE
    receipts = []
    for record in cohort['runs']:
        run = ROOT / 'runs' / record['run']
        if record['run_id'] in ['059', '061', '062']:
            # These complete local artifacts were independently verified by
            # Analysis019. The plotting pass also validates actual used arrays.
            assert json.loads((ROOT / 'analyses/019-2026-10-13-cifar10-eos-lr/results.json').read_text())['status'] == 'passed'
            receipts.append(verify_legacy(record))
            continue
        if not verified_local(record['run']):
            tool.REMOTE = old_remote if record.get('origin_analysis') == '020' else cohort['remote_root']
            tool.retrieve(record['run'])
        assert verified_local(record['run'])
        receipts.append(json.loads((run / 'retrieval_verification.json').read_text()))
    write_json(HERE / 'retrieval_verification.json', dict(status='passed', runs=receipts))
    print(json.dumps({'status': 'passed', 'conditions': len(receipts)}))


if __name__ == '__main__':
    main()
