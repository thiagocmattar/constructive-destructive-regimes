"""Compile the self-contained LaTeX report and check PDF structure/text bounds."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'report'
BUILD = ROOT / 'tmp/pdfs/analysis063-report-build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('pdflatex')
    if compiler is None:
        raise RuntimeError('pdflatex is required for the algorithm environment and report.')
    command = [compiler, '-interaction=nonstopmode', '-halt-on-error',
               '-output-directory=' + str(BUILD), 'instability-tracing-report.tex']
    for i in range(2):
        result = subprocess.run(command, cwd=REPORT, capture_output=True, text=True)
        (BUILD / f'compile-pass-{i+1}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'LaTeX compilation failed; see {BUILD}/compile-pass-{i+1}.txt')
    log = (BUILD / 'instability-tracing-report.log').read_text(encoding='utf-8', errors='replace')
    warnings = [line for line in log.splitlines() if 'Overfull' in line or 'undefined' in line or 'Rerun to get' in line]
    generated = BUILD / 'instability-tracing-report.pdf'
    document = fitz.open(generated)
    pages, overflow = [], []
    for i, page in enumerate(document):
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        for span in spans:
            x0, y0, x1, y1 = span['bbox']
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                overflow.append(dict(page=i+1, text=span['text'], bbox=list(span['bbox'])))
        pages.append(dict(page=i+1, characters=len(page.get_text()), spans=len(spans)))
    all_text = '\n'.join(p.get_text() for p in document)
    assert 'Algorithm 1' in all_text, 'Missing rendered algorithm environment'
    assert not overflow, overflow
    assert not warnings, warnings
    assert len(document) == 5, f'Expected five readable pages, found {len(document)}'
    assert all(p['characters'] > 500 for p in pages), 'Unexpected empty/fragment page'
    assert '??' not in all_text, 'Unresolved LaTeX reference'
    document.close()
    target = REPORT / generated.name
    shutil.copyfile(generated, target)
    inputs = [REPORT / 'instability-tracing-report.tex', REPORT / 'results-table.tex',
              *sorted((REPORT / 'figures').glob('*.pdf')), HERE / 'report_assets_verification.json']
    audit = dict(status='structural_checks_passed', pdf_sha256=sha(target), pages=pages,
        algorithm_environment=True, undefined_references=False, overfull_boxes=False, page_boundary_overflow=False,
        sources={p.relative_to(HERE).as_posix(): sha(p) for p in inputs}, generator_sha256=sha(Path(__file__)),
        visual_review='Required separately after rendering all five report pages and both MLP companion pages')
    (HERE / 'technical_report_verification.json').write_text(json.dumps(audit, indent=2) + '\n', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
