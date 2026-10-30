"""Hash inputs and source; separate source refreshes from expensive input staging."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',action='store_true');a=p.parse_args()
    if (HERE/'scratch/lease.json').exists():raise RuntimeError('Allocation freezes source packet')
    if a.inputs:
        packet=ROOT/'cache/runpod/analysis030-inputs.tar'
        if packet.exists():raise RuntimeError('Input packet already exists')
        sources=json.loads((HERE/'input_sources.json').read_text())
        with tarfile.open(packet,'w') as archive:
            for item in sources:
                path=Path(item['source']);path=path if path.is_absolute() else ROOT/path
                assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
                archive.add(path,arcname=item['path'])
        save(HERE/'input_packet.json',dict(path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,sha256=sha(packet)))
    files=[*sorted((ROOT/'src/baseline').glob('*.py')),ROOT/'tools/artifact_inventory.py',ROOT/'tests/test_lr_decisions.py']
    files+=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.sh'))+[HERE/'README.md',HERE/'cohort.json',HERE/'blocks.json',HERE/'probe.json',HERE/'input_sources.json']+sorted((HERE/'references').glob('*.json'))
    for row in json.loads((HERE/'cohort.json').read_text()):files.append(ROOT/'runs'/row['run']/'config.json')
    inventory=dict(files=[dict(path=f.relative_to(ROOT).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files])
    save(HERE/'source_inventory.json',inventory)
    packet=ROOT/'cache/runpod/analysis030-source.tar.gz'
    with tarfile.open(packet,'w:gz') as archive:
        for f in files+[HERE/'source_inventory.json']:archive.add(f,arcname=f.relative_to(ROOT).as_posix())
    save(HERE/'source_packet.json',dict(path=packet.relative_to(ROOT).as_posix(),bytes=packet.stat().st_size,sha256=sha(packet)))
    print('Packets ready')
if __name__=='__main__':main()
