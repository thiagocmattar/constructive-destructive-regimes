"""Check the completed reductions independently without regenerating figures."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    result = json.loads((HERE / 'results.json').read_text())
    cohort = json.loads((HERE / 'cohort.json').read_text())
    assert result['generator_sha256'] == digest(HERE / '06_analyze.py')
    assert result['run_order'] == [r['run'] for r in cohort]
    spec = importlib.util.spec_from_file_location('acceptance', HERE / '03_verify.py')
    accept = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accept)
    features, losses, validations, histories = [], [], [], []
    total_bytes = total_files = total_updates = total_tokens = 0
    finished = []
    for row, source in zip(cohort, result['sources']):
        root = ROOT / 'runs' / row['run']
        output = root / 'artifacts/attempt-01/full-pass'
        receipt = json.loads((root / 'prelaunch/attempt-01-retrieval.json').read_text())
        assert receipt['verified'] and receipt['scientific_verification']['verified']
        assert digest(output / 'inventory.json') == receipt['inventory_sha256'] == source['inventory_sha256']
        inventory = json.loads((output / 'inventory.json').read_text())
        for item in inventory['files']:
            assert (output / item['path']).stat().st_size == item['bytes']
        total_bytes += receipt['total_bytes']
        total_files += receipt['files']
        checked = accept.verify(output)
        total_updates += checked['updates']
        total_tokens += checked['input_tokens']
        for name in ['events', 'summary']:
            suffix = '.jsonl' if name == 'events' else '.json'
            assert digest(output / (name + suffix)) == source[name + '_sha256']
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['config_sha256'] == digest(root / 'config.json')
        finished.append(manifest['finished_at'])
        summary = json.loads((output / 'summary.json').read_text())
        updates = [r for r in map(json.loads, (output / 'events.jsonl').read_text().splitlines()) if r['event'] == 'update']
        histories.append(updates)
        probe = {r['local_update']: r['loss'] for r in summary['probes']}
        validation = {r['local_update']: r['loss'] for r in summary['validations']}
        validations.append(validation)
        losses.append(validation[128])
        early = updates[1:16]
        assert [r['local_update'] for r in early] == list(range(2, 17))
        config = json.loads((output / 'resolved_config.json').read_text())
        features.append([
            probe[0], probe[16] - probe[0], np.log(updates[0]['learning_rate']),
            np.log(config['model']['parameter_count']),
            sum(r['adjacent_cosine'] for r in early) / 15,
            sum(r['momentum']['adjacent_cosine'] for r in early) / 15,
            sum(r['adaptive']['adjacent_cosine'] for r in early) / 15,
            sum(r['incoming_previous_moment_cosine'] for r in early) / 15,
            np.log(sum(r['gradient_norm'] for r in early[-4:]) / sum(r['gradient_norm'] for r in early[:4])),
        ])
    x, y = np.array(features), np.array(losses)
    np.testing.assert_allclose(x, result['features'], rtol=0, atol=1e-12)
    predictions = {}
    seeds = np.array([r['seed'] for r in cohort])
    for label, count in [('Loss baseline', 4), ('Plus raw cosine', 5), ('Joint diagnostics', 9), ('Fixed LR', 0)]:
        predicted = np.empty(18)
        for seed in sorted(set(seeds)):
            train, test = np.flatnonzero(seeds != seed), np.flatnonzero(seeds == seed)
            if count:
                mean, scale = x[train, :count].mean(0), x[train, :count].std(0)
                scale[scale == 0] = 1
                a, b = (x[train, :count] - mean) / scale, (x[test, :count] - mean) / scale
                augmented = np.vstack([a, np.sqrt(10) * np.eye(count)])
                target = np.r_[y[train] - y[train].mean(), np.zeros(count)]
                beta = np.linalg.lstsq(augmented, target, rcond=None)[0]
                predicted[test] = b @ beta + y[train].mean()
            else:
                for i in test:
                    predicted[i] = np.mean([y[j] for j in train if cohort[j]['multiplier'] == cohort[i]['multiplier']])
        np.testing.assert_allclose(predicted, result['predictions'][label], rtol=0, atol=1e-12)
        predictions[label] = predicted
    effects = {(r['block'], r['multiplier']): r for r in result['effects']}
    for decision in result['decisions']:
        indices = [i for i, r in enumerate(cohort) if r['block'] == decision['block']]
        held = next(i for i in indices if cohort[i]['multiplier'] == 1)
        best = min(indices, key=lambda i: y[i])
        assert decision['best_multiplier'] == cohort[best]['multiplier']
        assert np.isclose(decision['held_regret'], y[held] - y[best], rtol=0, atol=1e-12)
        for i in indices:
            np.testing.assert_allclose(validations[i][0], validations[held][0], rtol=0, atol=1e-4)
            assert [r['batch_indices_sha256'] for r in histories[i]] == [r['batch_indices_sha256'] for r in histories[held]]
            effect = effects[cohort[i]['block'], cohort[i]['multiplier']]
            assert effect['block'] == cohort[i]['block'] and effect['multiplier'] == cohort[i]['multiplier']
            np.testing.assert_allclose([effect['final_loss'], effect['loss_difference_from_held']], [y[i], y[i] - y[held]], rtol=0, atol=1e-12)
            for key, column in [('early_raw_cosine', 4), ('early_moment_cosine', 5), ('early_adaptive_cosine', 6), ('early_input_moment_cosine', 7)]:
                assert np.isclose(effect[key], x[i, column], rtol=0, atol=1e-12)
        for label, pred in predictions.items():
            chosen = min(indices, key=lambda i: pred[i])
            recorded = decision['choices'][label]
            assert recorded['multiplier'] == cohort[chosen]['multiplier']
            assert np.isclose(recorded['regret'], y[chosen] - y[best], rtol=0, atol=1e-12)
    record = dict(verified=True, verified_at_epoch=time.time(), runs=18, updates=total_updates,
                  input_tokens=total_tokens, files=total_files, retained_bytes=total_bytes,
                  source_hashes_checked=True, artifact_hashes='Verified at retrieval; receipts, inventory hashes and current file sizes rechecked.',
                  feature_values_checked=int(x.size), held_seed_predictions_checked=72,
                  plotted_source_counts=dict(validation_points=54, scatter_points=36, regret_bars=30),
                  final_training_finished_at=max(finished), results_sha256=digest(HERE / 'results.json'),
                  verifier_sha256=digest(Path(__file__)))
    (HERE / 'result_verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
