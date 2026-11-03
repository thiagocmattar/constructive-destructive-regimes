"""Verify the selective restoration and build its separate review PDF."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
PREVIOUS = HERE.parent / '2026-11-03-regime-writing-revision'
SCRATCH = ROOT / 'tmp/pdfs/manuscript-sections-i-iv-restored-2026-11-03'

spec = importlib.util.spec_from_file_location('previous_verifier', PREVIOUS / '02_build_and_verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def check_restoration():
    manifest = previous.read(HERE / 'restoration_manifest.json')
    for group in ['restored_sources', 'retained_sources', 'preserved_previous_revision']:
        for path, digest in manifest[group].items():
            assert previous.sha(ROOT / path) == digest, (group, path)
            if group == 'restored_sources':
                baseline = subprocess.check_output(
                    ['git', 'show', manifest['baseline_commit'] + ':' + path], cwd=ROOT)
                assert (ROOT / path).read_bytes() == baseline, path
    return {group: len(manifest[group]) for group in
            ['restored_sources', 'retained_sources', 'preserved_previous_revision']}


def main():
    restoration = check_restoration()
    evidence, record = previous.check_sources()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tex = ['pdflatex', '--disable-installer', '-no-shell-escape', '-recorder',
           '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
           '-output-directory=' + str(SCRATCH), 'bare_jrnl.tex']
    commands = [tex, ['bibtex', str(SCRATCH / 'bare_jrnl')], tex, tex]
    for index, command in enumerate(commands, 1):
        result = subprocess.run(command, cwd=DRAFT, capture_output=True, timeout=120)
        (SCRATCH / f'build-{index}.txt').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'Build pass {index} failed; see {SCRATCH}')
    log = (SCRATCH / 'bare_jrnl.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if any(token in line.lower()
                for token in ['overfull', 'undefined', 'rerun to get', 'multiply defined'])]
    assert not problems, '\n'.join(problems)
    expected = {str((ROOT / f['target']).resolve()) for f in evidence['figures']}
    expected |= {str((ROOT / f['target']).resolve()) for f in
                 previous.read(previous.RETAINED / 'figure_manifest.json')['figures']}
    recorded = set()
    for line in (SCRATCH / 'bare_jrnl.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            p = Path(line[6:])
            p = p if p.is_absolute() else DRAFT / p
            if p.suffix == '.pdf' and p.resolve().is_relative_to((DRAFT / 'figures').resolve()):
                recorded.add(str(p.resolve()))
    assert expected == recorded and len(recorded) == 8
    pdf = SCRATCH / 'bare_jrnl.pdf'
    with fitz.open(pdf) as document:
        assert all('??' not in page.get_text() for page in document)
        record['pages'] = len(document)
    subprocess.run(['pdftoppm', '-r', '100', '-png', str(pdf), str(SCRATCH / 'review')],
                   check=True, capture_output=True, timeout=120)
    for target in [DRAFT / 'bare_jrnl.pdf', HERE / 'revised-manuscript.pdf']:
        shutil.copyfile(pdf, target)
    assert check_restoration() == restoration
    record.update(status='awaiting_visual_review', restoration=restoration,
                  pdf_sha256=previous.sha(pdf), active_figures=8,
                  verifier_sha256=previous.sha(__file__),
                  reused_verifier_sha256=previous.sha(PREVIOUS / '02_build_and_verify.py'),
                  restoration_manifest_sha256=previous.sha(HERE / 'restoration_manifest.json'),
                  build='pdflatex/BibTeX; zero overfull boxes or unresolved references')
    (HERE / 'verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in
                     ['status', 'pages', 'pdf_sha256', 'restoration', 'active_figures']}, indent=2))


if __name__ == '__main__':
    main()
