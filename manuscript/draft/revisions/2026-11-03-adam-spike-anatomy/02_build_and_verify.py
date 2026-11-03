"""Verify the Adam integration and retained evidence, then compile and render."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
SCRATCH = ROOT / 'tmp/pdfs/manuscript-adam-spike-anatomy-2026-11-03'
PRIOR = HERE.parent / '2026-11-02-feedback'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    evidence = read(HERE / 'evidence_manifest.json')
    assert evidence['generator_sha256'] == sha(HERE / '01_prepare_assets.py')
    for name, digest in evidence['sources'].items():
        assert sha(ROOT / name) == digest, name
    for figure in evidence['figures']:
        assert sha(ROOT / figure['source']) == sha(ROOT / figure['target']) == figure['sha256']
    table = ROOT / evidence['table']['path']
    assert sha(table) == evidence['table']['sha256']
    table_text = table.read_text(encoding='utf-8')
    assert '[' not in table_text and ']' not in table_text
    assert table_text.count(r'\multirow{3}') == 3
    for row in evidence['table']['rows']:
        assert ' & '.join(row['cells']) + r' \\' in table_text
    section = (DRAFT / 'sections/06-adam-diagnostics.tex').read_text(encoding='utf-8')
    assert section.index('fig:adam-overview') < section.index('fig:adam-spikes') < section.index('eq:adam-spike-boundaries')
    assert section.index('fig:adam-spike-distribution') < section.index('tab:adam-spike-anatomy') < section.index('eq:adam-event-product')
    for required in ['117 of 127', '21 of 127', '82 of 129', '858.8', '67--69', '90--94', '38--54',
        'pascanu2013difficulty', 'zhang2020clipping']:
        assert required in section, required
    for obsolete in ['analysis036_forecast', 'tab:adam-forecast', 'average precision', '16-state minimum', '0.454']:
        assert obsolete not in section
    spec = importlib.util.spec_from_file_location('retained_checks', PRIOR / '02_build_and_verify.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    record = dict(retained_math=previous.check_math(), retained_gd_sgd_evidence=previous.check_evidence(),
        adam_evidence_sha256=sha(HERE / 'evidence_manifest.json'))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tex = ['pdflatex', '--disable-installer', '-no-shell-escape', '-recorder',
        '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
        '-output-directory=' + str(SCRATCH), 'bare_jrnl.tex']
    for index, command in enumerate([tex, ['bibtex', str(SCRATCH / 'bare_jrnl')], tex, tex], 1):
        result = subprocess.run(command, cwd=DRAFT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH / f'build-{index}.txt').write_bytes(result.stdout)
        if result.returncode:
            print(result.stdout.decode(errors='replace')[-7000:])
            raise RuntimeError(f'Build pass {index} failed')
    log = (SCRATCH / 'bare_jrnl.log').read_text(errors='replace')
    for extra in range(3):
        if 'Rerun to get cross-references right' not in log:
            break
        result = subprocess.run(tex, cwd=DRAFT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH / f'build-extra-{extra + 1}.txt').write_bytes(result.stdout)
        assert result.returncode == 0
        log = (SCRATCH / 'bare_jrnl.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if any(token in line.lower()
        for token in ['overfull', 'undefined', 'rerun to get', 'multiply defined'])]
    if problems:
        print('\n'.join(problems))
        raise AssertionError('Resolve manuscript layout/reference warnings before acceptance')
    expected = {str((ROOT / f['target']).resolve()) for f in evidence['figures']}
    expected |= {str((ROOT / f['target']).resolve()) for f in read(PRIOR / 'figure_manifest.json')['figures']}
    expected.add(str((DRAFT / 'figures/analysis024/eos_gd_rho_3k.pdf').resolve()))
    recorded = set()
    for line in (SCRATCH / 'bare_jrnl.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            p = Path(line[6:])
            p = p if p.is_absolute() else DRAFT / p
            if p.suffix == '.pdf' and p.resolve().is_relative_to((DRAFT / 'figures').resolve()):
                recorded.add(str(p.resolve()))
    assert recorded == expected and len(recorded) == 9
    pdf = SCRATCH / 'bare_jrnl.pdf'
    with fitz.open(pdf) as document:
        text = '\n'.join(page.get_text() for page in document)
        assert '??' not in text and 'Average precision for impending' not in text
        record['pages'] = len(document)
        auxiliary = (SCRATCH / 'bare_jrnl.aux').read_text()
        bounds = [int(re.search(r'\\newlabel\{' + label + r'\}\{\{[^}]+\}\{(\d+)\}', auxiliary).group(1))
            for label in ['sec:adam-diagnostics', 'sec:conclusion']]
        record['adam_pages'] = list(range(bounds[0], bounds[1] + 1))
    result = subprocess.run(['pdftoppm', '-r', '100', '-png', str(pdf), str(SCRATCH / 'review')],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    for target in [DRAFT / 'bare_jrnl.pdf', HERE / 'revised-manuscript.pdf']:
        shutil.copyfile(pdf, target)
    record.update(status='awaiting_visual_review', pdf_sha256=sha(pdf), active_figures=9,
        build='pdflatex, bibtex, pdflatex, pdflatex; zero overfull boxes or unresolved references',
        verifier_sha256=sha(__file__),
        source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in [DRAFT / 'bare_jrnl.tex',
            DRAFT / 'bibliography.bib', *sorted((DRAFT / 'sections').glob('*.tex')), table]})
    (HERE / 'verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
