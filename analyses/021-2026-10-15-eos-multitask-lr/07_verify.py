"""Verify matched factors, frozen identities, indexing and manuscript placement."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json


def module(name):
    spec = importlib.util.spec_from_file_location('verify_' + name, HERE / name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    old = json.loads((ROOT / 'analyses/020-2026-10-14-eos-multitask/cohort.json').read_text())
    new = [r for r in cohort['runs'] if r['kind'] == 'new_training']
    assert len(new) == 16 and len({r['run_id'] for r in cohort['runs']}) == 27
    assert sorted(int(r['run_id']) for r in new) == list(range(104, 120))
    for task, rates in cohort['rates'].items():
        for lr in rates:
            group = [r for r in cohort['runs'] if r['task'] == task and r['lr'] == lr]
            assert sorted(r['seed'] for r in group) == [0, 1, 2]
            assert all(r['method'] == 'gd' and r['updates'] == 500 for r in group)
    for r in new:
        base = next(d for d in old['runs'] if d['method'] == 'gd' and d['task'] == r['task'] and d['seed'] == r['seed'])
        a, b = copy.deepcopy(r['config']), copy.deepcopy(base['config'])
        for key in ['run_id', 'hypothesis', 'lr', 'notes']:
            a.pop(key); b.pop(key)
        assert a == b, 'An unmatched scientific factor changed.'
    for item in json.loads((HERE / 'source_inventory.json').read_text())['files']:
        p = ROOT / item['path']
        assert p.stat().st_size == item['bytes'] and file_hash(p) == item['sha256']
    packet = json.loads((HERE / 'packet.json').read_text())
    assert file_hash(ROOT / packet['path']) == packet['sha256']
    plot, manuscript, collect = module('04_plot.py'), module('05_manuscript.py'), module('03_collect.py')
    for r in cohort['runs']:
        if r['run_id'] in ['059', '061', '062']:
            collect.verify_legacy(r)
            d = plot.load(r)
            assert d['x'][0] == 1 and d['x'][-1] == 500
            assert d['q'][1:].shape == (499,) and len(d['spectral_x']) == 6
    # An independent quadratic GD identity catches sign or LR-factor mistakes.
    hessian = np.array([[3., .4], [.4, 1.]])
    for lr in [.004, .02, .1]:
        x = np.array([1.2, -.7]); g0 = hessian @ x
        g1 = hessian @ (x - lr * g0)
        q = 1 - g0.dot(g1) / g0.dot(g0)
        np.testing.assert_allclose(q, lr * g0.dot(hessian @ g0) / g0.dot(g0), atol=1e-14)
    source = (ROOT / 'manuscript/2026-10-11-restatement/gd-formulation.tex').read_bytes().decode()
    inserted = manuscript.insert(source)
    marker = '\n\n\\input{../../analyses/' + HERE.name + '/figure.tex}'
    assert inserted.replace(marker, '', 1) == source
    assert manuscript.insert(inserted) == inserted
    table_end = inserted.index('\\end{table}', inserted.index('\\label{tab:sign}')) + len('\\end{table}')
    assert inserted[table_end:].startswith(marker)
    assert not (HERE / 'figures/eos_multitask_learning_rates.pdf').exists(), 'Do not mistake a layout preview for completed evidence.'
    write_json(HERE / 'verification.json', dict(status='passed',
        checks=['27 conditions, 16 distinct new IDs, three seeds at every rate',
            'Only LR/run identity/notes differ from matched Analysis020 full-GD configs',
            'All scientific source/config/packet hashes verified',
            '1497 actual MLP gradient pairs satisfy the independently recomputed cosine/norm identity',
            'All 45 retained files of the reused legacy MLP runs rehashed against their transfer inventories',
            'Exact quadratic GD sign and LR scaling checked independently',
            'Loss and spectral endpoints/cadence checked on real prefixes',
            'Insertion follows tab:sign, preserves all surrounding source and is idempotent',
            'No unfinished multitask figure exists; layout checks stay in ignored scratch'],
        cohort_sha256=file_hash(HERE / 'cohort.json'),
        plot_script_sha256=file_hash(HERE / '04_plot.py'), manuscript_script_sha256=file_hash(HERE / '05_manuscript.py')))
    print('All matching, source, metric and insertion checks passed.')


if __name__ == '__main__':
    main()
