"""Retrieve only terminal calibration evidence through the qualified hash verifier."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import write_json


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    tools = module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'calibration_retrieval_checks')
    retriever = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/05_retrieve.py', 'calibration_retriever')
    retriever.REMOTE = cohort['remote_root']
    receipts = []
    for run in cohort['runs']:
        if run['kind'] == 'reused_gd_prefix':
            receipts.append(tools.verify_legacy(run))
        else:
            if not tools.verified_local(run['run']):
                retriever.retrieve(run['run'])
            assert tools.verified_local(run['run'])
            receipts.append(json.loads((ROOT / 'runs' / run['run'] / 'retrieval_verification.json').read_text()))
    write_json(HERE / 'retrieval_verification.json', dict(status='passed', runs=receipts))
    print(json.dumps(dict(status='passed', conditions=len(receipts))))


if __name__ == '__main__':
    main()
