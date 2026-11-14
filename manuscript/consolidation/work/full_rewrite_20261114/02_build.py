"""Build the complete paper and supplement; render every page for visual review."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import fitz

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DRAFT = BASE / 'draft'
BUILD = HERE / 'scratch/latex'
BUILD.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--main-only', action='store_true')
args = parser.parse_args()


def run(command, output):
    result = subprocess.run(command, cwd=DRAFT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (BUILD / output).write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-6000:])


previous = HERE / 'build.json'
results = json.loads(previous.read_text(encoding='utf-8')) if args.main_only and previous.exists() else {}
for name in ['manuscript'] + ([] if args.main_only else ['supplementary']):
    command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-recorder', f'-output-directory={BUILD}', f'{name}.tex']
    run(command, f'{name}-pass1.txt')
    run(['bibtex', str(BUILD / name)], f'{name}-bibtex.txt')
    for i in [2, 3]:
        run(command, f'{name}-pass{i}.txt')
    log = (BUILD / f'{name}.log').read_text(encoding='utf-8', errors='replace')
    warnings = [line.strip() for line in log.splitlines() if re.search(r'Overfull|Underfull|undefined|multiply defined|LaTeX Warning|Package .* Warning', line)]
    target = DRAFT / f'{name}.pdf'
    shutil.copyfile(BUILD / target.name, target)
    render = HERE / f'scratch/review/{name}'
    render.mkdir(parents=True, exist_ok=True)
    with fitz.open(target) as doc:
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(render / f'page-{i+1:02}.png')
        results[name] = {'pages': len(doc), 'warnings': warnings,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'characters': sum(len(p.get_text()) for p in doc)}
(HERE / 'build.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(results, indent=2))
