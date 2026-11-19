"""Verify the prior evidence and package just the inputs needed on the GPU."""
import hashlib
import json
import tarfile
from pathlib import Path

import torch
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, tensor_hash, write_json
from src.eos_tasks.task import model_for

old = next((ROOT / 'runs').glob('216-*'))
baseline = next((ROOT / 'runs').glob('125-*'))
cache = ROOT / 'cache/eos-multitask'
work = HERE / 'artifacts'
work.mkdir(exist_ok=True)
cfg = json.loads((old / 'config.json').read_text())
cfg.update(run_id='298', seeds=[0], learning_rates={'transformer': 0.1},
           reference_policy='Pinned seed-0 initialization and Run125 state after 2000 GD updates',
           notes=['Reuse Run216 sampling and accumulation with a single initialization.'])
write_json(HERE / 'config.json', cfg)
historical = json.loads((old / 'input_manifest.json').read_text())['files']
inputs = []
for item in historical:
    name = item['path']
    if name.startswith(('runs/216', 'runs/214')) and not name.endswith('214-2026-11-02-sgd-batch-extension/01_measure.py'):
        continue
    if 'transformer' not in item['tasks'] or 'initial_seed1' in name or 'initial_seed2' in name:
        continue
    assert file_hash(ROOT / name) == item['sha256'], name
    inputs.append(item)
shared = json.loads((cache / 'inputs.json').read_text())
original = baseline / 'artifacts/attempt-01-spark'
inventory = json.loads((original / 'transfer_inventory.json').read_text())['files']
checkpoint = original / 'production/state-002000.pt'
expected = next(f['sha256'] for f in inventory if f['path'] == 'production/state-002000.pt')
assert file_hash(checkpoint) == expected
trained = torch.load(checkpoint, weights_only=False, map_location='cpu')
assert trained['step'] == 2000 and trained['config_sha256'] == file_hash(baseline / 'config.json')
assert trained['inputs_sha256'] == file_hash(cache / 'inputs.json')
model = model_for('transformer', cache, shared['wikitext']['vocabulary_size'])
initial = cache / shared['initial']['transformer'][0]['path']
sources = {}
for step, state, source in [(0, torch.load(initial, weights_only=True), initial),
                            (2000, trained['model'], checkpoint)]:
    model.load_state_dict(state)
    digest = tensor_hash(model.parameters())
    if step == 0:
        assert digest == shared['initial']['transformer'][0]['initial_tensor_sha256']
    path = work / f'state-{step:04}.pt'
    torch.save(dict(model=state, tensor_sha256=digest, state=step, task='transformer',
                    seed=0, mechanics_only=False, config_sha256=file_hash(HERE / 'config.json')), path)
    sources[str(step)] = dict(path=source.relative_to(ROOT).as_posix(),
                              sha256=file_hash(source), tensor_sha256=digest)
    inputs.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                       sha256=file_hash(path), tasks=['transformer']))
for path in (HERE / '01_run.py', HERE / 'config.json'):
    inputs.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                       sha256=file_hash(path), tasks=['transformer']))
manifest = HERE / 'input_manifest.json'
write_json(manifest, dict(run_id='298', files=inputs, sources=sources))
with tarfile.open(work / 'worker.tar', 'w') as archive:
    for name in [x['path'] for x in inputs] + [manifest.relative_to(ROOT).as_posix()]:
        archive.add(ROOT / name, arcname=name)
print('Verified worker archive', file_hash(work / 'worker.tar'), flush=True)
