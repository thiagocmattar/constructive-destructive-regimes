"""Verify pinned local inputs and write Run213's portable byte/SHA manifest."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json

RUN = ROOT/'runs/213-2026-11-02-eos-adam-from-init-calibration'
CACHE = ROOT/'cache/eos-multitask'
MLP = ROOT/'cache/cifar10-eos'


def main():
    cfg = json.loads((RUN/'config.json').read_text())
    shared = json.loads((CACHE/'inputs.json').read_text())
    old = json.loads((MLP/'inputs.json').read_text())
    if file_hash(CACHE/'inputs.json') != 'ef6f5b79249c05b1deda802d8b24c55e186c75825bc80f21a0c6f66c829d729f':
        raise RuntimeError('Analysis024 input manifest changed.')
    files = []
    def add(file, tasks, expected=None):
        sha = file_hash(file)
        if expected is not None and sha != expected:
            raise RuntimeError('Pinned file changed: '+str(file))
        files.append(dict(path=file.relative_to(ROOT).as_posix(), bytes=file.stat().st_size, sha256=sha, tasks=tasks))
    add(CACHE/'inputs.json', cfg['tasks'])
    add(MLP/'inputs.json', ['mlp'])
    add(MLP/'data.pt', ['mlp'], old['prepared_files']['data.pt']['sha256'])
    for task in cfg['tasks']:
        initial = shared['initial'][task][cfg['seed']]
        add(CACHE/initial['path'], [task], initial['sha256'])
        if task != 'mlp':
            file = task+'_data.pt'
            add(CACHE/file, [task], shared['prepared_files'][file]['sha256'])
    for name, identity in shared['reference_sources'].items():
        add(CACHE/'reference'/name, cfg['tasks'], identity['sha256'])
    for folder in ['src/eos_tasks', 'src/cifar_eos']:
        for file in sorted((ROOT/folder).glob('*.py')):
            add(file, cfg['tasks'])
    for name in ['01_train.py', '02_classify.py', 'diagnostics.py', 'config.json']:
        add(RUN/name, cfg['tasks'])
    result = dict(run_id='213', status='locally_verified_not_launched',
        files=files, initial=shared['initial'], wikitext=shared['wikitext'],
        limitations=['Seed0 screening only.', 'No production-shaped remote throughput, price, spending authority or launch yet.'])
    write_json(RUN/'input_manifest.json', result)
    print(json.dumps(dict(files=len(files), bytes=sum(x['bytes'] for x in files), status=result['status'])))


if __name__ == '__main__':
    main()
