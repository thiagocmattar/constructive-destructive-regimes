"""Independent known-effect fixtures for H021 scalar reduction; no cloud calls."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'analyses/025-2026-10-17-pythia-descent-mechanism/08_analyze.py'


@pytest.fixture
def cohort(tmp_path):
    spec = importlib.util.spec_from_file_location('causal_analysis', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.HERE = tmp_path / 'analysis'
    module.HERE.mkdir()
    records = []
    for seed in module.SEEDS:
        for i, size in enumerate(module.SIZES):
            baseline = f'{size}-{seed}-baseline'
            records.append(dict(run=baseline, condition='baseline', size=size, seed=seed))
            native = [dict(candidate='native_adamw', update=t, loss_after=10-t*.001)
                      for t in [100,150,175,200,225,250,275,300,350]]
            write_run(module, baseline, [dict(event='initial_probe', loss=10)], native, {})
            for fork in [175,225]:
                name = f'{size}-{seed}-fork{fork}'
                records.append(dict(run=name, condition='ablation', size=size, seed=seed,
                                    parent=baseline, junction=fork))
                known_tau = .01 * (i+1)
                write_run(module, name, [dict(event='initial_probe', loss=10-fork*.001)], [],
                          dict(final_probe_loss=(10-(fork+50)*.001)*math.exp(known_tau)))
    module.save(module.HERE / 'cohort.json', records)
    module.save(module.HERE / 'retrieved.json', dict(runs=[r['run'] for r in records]))
    return module


def write_run(module, name, events, probes, summary):
    run = module.ROOT / 'runs' / name
    output = run / 'artifacts/attempt-01/full-pass'
    output.mkdir(parents=True)
    for filename, records in [('events.jsonl', events), ('probe_events.jsonl', probes)]:
        (output / filename).write_text(''.join(json.dumps(r)+'\n' for r in records))
    module.save(output / 'summary.json', summary)
    inventory = dict(files=[dict(path=p.name, bytes=p.stat().st_size, sha256=module.digest(p))
                           for p in output.iterdir()])
    module.save(output / 'inventory.json', inventory)
    (run / 'prelaunch').mkdir()
    module.save(run / 'prelaunch/attempt-01-retrieval.json', dict(verified=True,
        scientific_verification=dict(verified=True), inventory_sha256=module.digest(output / 'inventory.json')))


def test_primary_effect_and_paired_size_interaction(cohort):
    result = cohort.reduce()
    assert len(result['branch_effects']) == 18
    assert len(result['size_interactions']) == 6
    for row in result['branch_effects']:
        assert row['tau'] == pytest.approx(.01 * (cohort.SIZES.index(row['size'])+1))
        assert row['ablated_log_loss_change']-row['native_log_loss_change'] == pytest.approx(row['tau'])
    for row in result['size_interactions']:
        assert row['tau70_minus_tau14'] == pytest.approx(.02)
        assert row['tau31_minus_tau14'] == pytest.approx(.01)


def test_changed_scalar_evidence_is_rejected(cohort):
    output = next((cohort.ROOT / 'runs').iterdir()) / 'artifacts/attempt-01/full-pass'
    with (output / 'events.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(AssertionError):
        cohort.reduce()


def test_incomplete_cohort_is_rejected(cohort):
    cohort.save(cohort.HERE / 'retrieved.json', dict(runs=[]))
    with pytest.raises(AssertionError, match='full cohort'):
        cohort.reduce()
