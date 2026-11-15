"""Build only the revised main manuscript; preserve all other paper inputs."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import fitz

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / 'draft'
BUILD = HERE / 'scratch/latex'
BUILD.mkdir(parents=True, exist_ok=True)


def run(command, name):
    result = subprocess.run(command, cwd=DRAFT, capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    (BUILD / name).write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-6000:])


command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-recorder',
           f'-output-directory={BUILD}', 'manuscript.tex']
run(command, 'pass1.txt')
run(['bibtex', str(BUILD / 'manuscript')], 'bibtex.txt')
for i in (2, 3):
    run(command, f'pass{i}.txt')
log = (BUILD / 'manuscript.log').read_text(encoding='utf-8', errors='replace')
warnings = [line.strip() for line in log.splitlines()
            if re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', line)]
if warnings:
    raise RuntimeError('\n'.join(warnings))
target = DRAFT / 'manuscript.pdf'
shutil.copyfile(BUILD / target.name, target)
render = HERE / 'scratch/review'
render.mkdir(exist_ok=True)
with fitz.open(target) as doc:
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(render / f'page-{i+1:02}.png')
    result = {'pages': len(doc), 'warnings': warnings,
              'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
(HERE / 'build.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
