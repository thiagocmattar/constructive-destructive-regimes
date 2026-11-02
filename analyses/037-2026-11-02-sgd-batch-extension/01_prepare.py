"""Freeze verified local scientific inputs; no training or cloud allocation."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,write_json
RUN=ROOT/'runs/214-2026-11-02-sgd-batch-extension'


def main():
    if any((RUN/'artifacts').rglob('manifest.json')):
        raise RuntimeError('Production has started; do not rewrite its input identity.')
    cfg=json.loads((RUN/'config.json').read_text()); cache=ROOT/cfg['cache']
    shared=json.loads((cache/'inputs.json').read_text())
    assert file_hash(cache/'inputs.json')=='ef6f5b79249c05b1deda802d8b24c55e186c75825bc80f21a0c6f66c829d729f'
    files=[]
    def add(path,tasks,expected=None):
        digest=file_hash(path)
        if expected: assert digest==expected,str(path)
        files.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=digest,tasks=tasks))
    add(cache/'inputs.json',cfg['tasks'])
    for task in cfg['tasks']:
        name=task+'_data.pt'; add(cache/name,[task],shared['prepared_files'][name]['sha256'])
        for seed in cfg['seeds']:
            initial=shared['initial'][task][seed]
            add(cache/initial['path'],[task],initial['sha256'])
    for name,entry in shared['reference_sources'].items():
        add(cache/'reference'/name,cfg['tasks'],entry['sha256'])
    for folder in ['src/eos_tasks','src/cifar_eos']:
        for path in sorted((ROOT/folder).glob('*.py')): add(path,cfg['tasks'])
    for name in ['config.json','01_measure.py']: add(RUN/name,cfg['tasks'])
    write_json(RUN/'input_manifest.json',dict(run_id='214',files=files,initial=shared['initial'],
        wikitext=shared['wikitext'],reference_policy=cfg['reference_policy']))
    print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files),status='verified')))


if __name__=='__main__':main()
