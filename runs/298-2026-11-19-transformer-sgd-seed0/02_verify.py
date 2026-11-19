"""Independently reduce the 66 downloaded arms and check the sampling and geometry."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    cfg = read(HERE / 'config.json')
    inputs = read(HERE / 'input_manifest.json')
    config_hash = digest(HERE / 'config.json')
    input_hash = digest(HERE / 'input_manifest.json')
    directory = HERE / 'artifacts/results'
    batches = cfg['batches']['transformer']
    full = batches[-1]
    output = []
    retained_bytes = 0
    inventory_files = 0
    starting_losses = {}
    for state in cfg['states']:
        start_file = HERE / f'artifacts/state-{state:04d}.pt'
        start = torch.load(start_file, weights_only=False, map_location='cpu')
        assert start['tensor_sha256'] == inputs['sources'][str(state)]['tensor_sha256']
        source_hash = digest(start_file)
        for batch in batches:
            repetitions = 1 if batch == full else cfg['stochastic_repetitions']
            for rep in range(repetitions):
                job = directory / f'state-{state}-b{batch}-rep{rep:02d}'
                inventory = read(job / 'inventory.json')
                for filename, metadata in inventory.items():
                    target = job / filename
                    assert target.stat().st_size == metadata['bytes'] and digest(target) == metadata['sha256']
                    retained_bytes += metadata['bytes']
                    inventory_files += 1
                summary = read(job / 'summary.json')
                identity = summary['identity']
                assert (identity['task'], identity['mode'], identity['seed'], identity['state'],
                        identity['batch'], identity['repetition'], identity['mechanics_only']) == (
                            'transformer', 'fork', 0, state, batch, rep, False)
                assert identity['config_sha256'] == config_hash and identity['input_manifest_sha256'] == input_hash
                assert identity['source']['sha256'] == source_hash
                assert read(job / 'manifest.json')['starting_tensor_sha256'] == start['tensor_sha256']
                assert read(job / 'resolved_config.json') == cfg and summary['status'] == 'complete'
                history = [json.loads(line) for line in (job / 'metrics.jsonl').read_text().splitlines()]
                assert [r['update'] for r in history] == list(range(1, 201))
                assert history[0]['cosine'] is None and summary['valid_cosines'] == 199
                cosines = np.asarray([r['cosine'] for r in history[1:]])
                assert np.isfinite(cosines).all() and np.abs(cosines).max() <= 1 + 1e-8
                np.testing.assert_allclose(cosines.mean(), summary['mean_rho'], atol=1e-12, rtol=0)
                assert np.isfinite([summary['initial_loss'],summary['final_loss']]).all()
                np.testing.assert_allclose(summary['initial_loss'], starting_losses.setdefault(state, summary['initial_loss']), rtol=1e-6)
                if batch != full:
                    rng = np.random.default_rng(np.random.SeedSequence([cfg['sampling_seed'], 2, 0, state, rep]))
                    draws = np.fromfile(job / 'indices.u32', dtype='<u4').reshape(200, batch)
                    for draw in draws:
                        np.testing.assert_array_equal(draw, rng.permutation(full)[:batch])
                else:
                    assert (job / 'indices.u32').stat().st_size == 0
                finish = torch.load(job / 'final.pt', weights_only=False, map_location='cpu')
                assert finish['state'] == state + 200
                squared = sum((finish['model'][key].double() - value.double()).square().sum().item()
                              for key, value in start['model'].items() if value.is_floating_point())
                np.testing.assert_allclose(squared ** 0.5, summary['absolute_displacement'], rtol=1e-8)
                assert summary['max_actual_prefix_error'] < 1e-10
                if rep == 0:
                    audit = torch.load(job / 'audit-0199.pt', weights_only=False, map_location='cpu')
                    g = audit['gradient'].double()
                    prev = audit['previous_gradient'].double()
                    measured = (g @ prev / (g.norm() * prev.norm())).item()
                    np.testing.assert_allclose(measured, cosines[-1], atol=1e-10)
                output.append(dict(state=state, batch=batch, repetition=rep, mean_rho=float(cosines.mean()),
                                   initial_loss=summary['initial_loss'], final_loss=summary['final_loss'],
                                   inventory_sha256=digest(job / 'inventory.json')))
    assert len(output) == 66 and len(list(directory.glob('state-*/summary.json'))) == 66
    out = dict(verified_arms=len(output), updates=13200, cosine_pairs=13134, inventory_files=inventory_files,
               retained_bytes=retained_bytes, config_sha256=config_hash, inputs_sha256=input_hash, arms=output)
    (HERE / 'verified.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
