"""Verify the historical framing and main text; build its separate review PDF."""
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
PREVIOUS = HERE.parent / '2026-11-03-regime-writing-revision'
SCRATCH = ROOT / 'tmp/pdfs/manuscript-original-argument-restored-2026-11-03'

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
    front = manifest['restored_front_matter']
    assert previous.sha(ROOT / front['path']) == front['after_sha256']
    old_front = subprocess.check_output(
        ['git', 'show', manifest['baseline_commit'] + ':' + front['path']], cwd=ROOT)
    current_front = (ROOT / front['path']).read_bytes()
    for pattern in [rb'^\\title\{[^\n]*\}\r?$',
                    rb'\\begin\{abstract\}.*?\\end\{abstract\}',
                    rb'\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}']:
        assert re.search(pattern, current_front, re.S | re.M).group(0) == (
            re.search(pattern, old_front, re.S | re.M).group(0))
    introduction = (DRAFT / 'sections/01-introduction.tex').read_text(encoding='utf-8')
    assert introduction.split('\\section{Introduction}', 1)[1].lstrip().startswith(
        manifest['expected_introduction_opening'])
    return {group: len(manifest[group]) for group in
            ['restored_sources', 'retained_sources', 'preserved_previous_revision']}


def check_evidence():
    evidence = previous.read(PREVIOUS / 'evidence_manifest.json')
    assert evidence['generator_sha256'] == previous.sha(PREVIOUS / '01_prepare_table.py')
    for path, digest in evidence['sources'].items():
        assert previous.sha(ROOT / path) == digest, path
    for figure in evidence['figures']:
        assert previous.sha(ROOT / figure['source']) == previous.sha(ROOT / figure['target']) == figure['sha256']
    assert previous.sha(ROOT / evidence['table']['path']) == evidence['table']['sha256']
    spec = importlib.util.spec_from_file_location(
        'retained_evidence', previous.RETAINED / '02_build_and_verify.py')
    retained = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retained)
    return evidence, dict(retained_math=retained.check_math(),
                          additional_math=previous.extra_math(),
                          retained_gd_sgd_evidence=retained.check_evidence(),
                          adam_claims=evidence['claims'])


def main():
    restoration = check_restoration()
    evidence, record = check_evidence()
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
    for extra in range(3):
        if 'Rerun to get cross-references right' not in log:
            break
        result = subprocess.run(tex, cwd=DRAFT, capture_output=True, timeout=120)
        (SCRATCH / f'build-extra-{extra+1}.txt').write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0
        log = (SCRATCH / 'bare_jrnl.log').read_text(errors='replace')
    manifest = previous.read(HERE / 'restoration_manifest.json')
    missing = sorted(set(re.findall(r"Reference `([^']+)'[^\n]*undefined", log)))
    assert set(missing) <= set(manifest['allowed_unresolved_labels']), missing
    problems = [line for line in log.splitlines() if any(token in line.lower()
                for token in ['overfull', 'rerun to get', 'multiply defined'])
                or ('undefined' in line.lower() and 'Reference `' not in line
                    and 'There were undefined references' not in line)]
    assert not problems, '\n'.join(problems)
    record['known_broken_references'] = missing
    record['reference_policy'] = manifest['reference_policy']
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
        record['pages'] = len(document)
        first_page = ' '.join(document[0].get_text().split())
        assert first_page.split('I. INTRODUCTION', 1)[1].lstrip().startswith(
            manifest['expected_introduction_opening']), first_page
        record['introduction_opening_verified_in_pdf'] = manifest['expected_introduction_opening']
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
                  build='pdflatex/BibTeX; zero overfull boxes; author-accepted broken references recorded')
    (HERE / 'verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in
                     ['status', 'pages', 'pdf_sha256', 'restoration', 'active_figures',
                      'introduction_opening_verified_in_pdf', 'known_broken_references']}, indent=2))


if __name__ == '__main__':
    main()
