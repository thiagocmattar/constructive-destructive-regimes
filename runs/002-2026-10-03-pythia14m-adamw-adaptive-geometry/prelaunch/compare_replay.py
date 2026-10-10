"""Quantify raw-stream replay agreement without assuming bitwise determinism."""
import hashlib
import json
from pathlib import Path
import statistics
from safetensors.torch import load_file

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
OLD = REPO / 'runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/artifacts/attempt-04/full-pass'
NEW = RUN / 'artifacts/attempt-02/full-pass'

def read(root, name):
    return json.loads((root/name).read_text())

def events(root):
    return [e for line in (root/'events.jsonl').read_text().splitlines()
            if (e := json.loads(line))['event'] == 'update']

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

old, new = events(OLD), events(NEW)
assert len(old) == len(new) == 712
c1, c2 = [read(p, 'resolved_config.json') for p in (OLD, NEW)]
for section in ('data', 'seeds', 'training', 'validation', 'checkpoints'):
    assert c1[section] == c2[section], section
assert all(c1['model'][k] == c2['model'][k] for k in c1['model'])
assert read(OLD,'input_identity.json')['order_sha256'] == read(NEW,'input_identity.json')['order_sha256']
initial_hashes = []
names = read(NEW, 'gradient_state_identity.json')['parameter_names']
assert names == read(OLD, 'gradient_state_identity.json')['parameter_names']
for root in (OLD, NEW):
    parameters = load_file(root/'checkpoints/initial/model/model.safetensors')
    initial = hashlib.sha256()
    for name in names:
        initial.update(name.encode() + b'\0')
        initial.update(parameters[name].numpy().tobytes())
    initial_hashes.append(initial.hexdigest())
assert initial_hashes[0] == initial_hashes[1] == c2['model']['initial_parameter_sha256']
results = {}
for key in ('loss','gradient_norm','adjacent_cosine','cumulative_norm_sum',
            'cumulative_gradient_norm','cumulative_ratio'):
    pairs = [(a[key],b[key]) for a,b in zip(old,new) if a[key] is not None and b[key] is not None]
    differences = [abs(a-b) for a,b in pairs]
    results[key] = {'defined_pairs':len(pairs),'median_absolute_difference':statistics.median(differences),
                    'maximum_absolute_difference':max(differences),
                    'last_absolute_difference':differences[-1]}
result = {'matched_scientific_factors':True,
          'initial_checkpoint_parameters_identical':True,
          'initial_parameter_sha256':read(NEW,'environment.json')['initial_parameter_sha256'],
          'source_event_hashes':{'run001':digest(OLD/'events.jsonl'),'run002':digest(NEW/'events.jsonl')},
          'raw_stream_agreement':results,
          'validation_loss_absolute_difference':abs(read(OLD,'summary.json')['final_validation']['loss']-
                                                    read(NEW,'summary.json')['final_validation']['loss']),
          'interpretation':'Matched recipe and initial weights; numerical replay differences are descriptive, not a bitwise-determinism claim.'}
(RUN/'prelaunch/replay-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
