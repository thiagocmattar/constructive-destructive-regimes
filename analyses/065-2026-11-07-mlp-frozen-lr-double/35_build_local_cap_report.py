"""Compile the cap-only technical report; check structure, references and bounds."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'report'
BUILD = ROOT / 'tmp/pdfs/analysis065-local-cap-build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('pdflatex')
    assert compiler, 'pdflatex required'
    assets = json.loads((HERE / 'local_cap_report_assets.json').read_text())
    assert assets['status'] == 'verified'
    assert assets['generator_sha256'] == sha(HERE / '34_local_cap_report_assets.py')
    for name, digest in assets['sources'].items():
        assert sha(HERE / name) == digest
    for output in assets['outputs']:
        assert sha(HERE / output['file']) == output['sha256']
    for name, digest in assets['table_sources'].items():
        assert sha(REPORT / name) == digest
    command = [compiler, '-interaction=nonstopmode', '-halt-on-error',
               '-output-directory=' + str(BUILD), 'local-displacement-cap-report.tex']
    for i in range(2):
        result = subprocess.run(command, cwd=REPORT, capture_output=True, text=True)
        (BUILD / f'compile-pass-{i + 1}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == 0, f'Compilation failed: {BUILD}/compile-pass-{i+1}.txt'
    log = (BUILD / 'local-displacement-cap-report.log').read_text(encoding='utf-8', errors='replace')
    warnings = [line for line in log.splitlines() if any(s in line for s in ['Overfull', 'undefined', 'Rerun to get', 'multiply defined'])]
    generated = BUILD / 'local-displacement-cap-report.pdf'
    document = fitz.open(generated)
    pages, overflow = [], []
    for i, page in enumerate(document):
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        for span in spans:
            x0, y0, x1, y1 = span['bbox']
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                overflow.append(dict(page=i+1, text=span['text'], bbox=list(span['bbox'])))
        pages.append(dict(page=i+1, characters=len(page.get_text()), spans=len(spans)))
    text = '\n'.join(p.get_text() for p in document)
    assert 'Algorithm 1' in text and '??' not in text
    assert not overflow, overflow
    assert not warnings, warnings
    assert len(document) == 7, f'Expected seven readable pages, found {len(document)}'
    assert all(p['characters'] > 500 for p in pages)
    document.close()
    target = REPORT / generated.name
    shutil.copyfile(generated, target)
    inputs = [REPORT / 'local-displacement-cap-report.tex', *sorted(REPORT.glob('cap-*-table.tex')),
              HERE / 'local_cap_report_assets.json', *[HERE / o['file'] for o in assets['outputs']]]
    audit = dict(status='structural_checks_passed', pdf_sha256=sha(target), pages=pages,
        algorithm_environment=True, undefined_references=False, overfull_boxes=False, page_boundary_overflow=False,
        sources={p.relative_to(HERE).as_posix(): sha(p) for p in inputs}, generator_sha256=sha(Path(__file__)),
        visual_review='Required after rendering the seven report pages and three standalone figure pages.')
    (HERE / 'local_cap_report_verification.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pages': len(pages), 'warnings': warnings, 'pdf': str(target)}))


if __name__ == '__main__':
    main()
