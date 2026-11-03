"""Check mathematical identities, frozen evidence and authoring; build and render."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRAFT = HERE.parents[1]
SCRATCH = ROOT / 'tmp/pdfs/manuscript-regime-writing-2026-11-03'
RETAINED = HERE.parent / '2026-11-02-feedback'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extra_math():
    rng = np.random.default_rng(20261103)
    errors, checks = [], 0
    # General adaptive displacement: the projected response is bilinear
    # in the old raw gradient and the actual adaptive direction.
    for _ in range(40):
        a = rng.normal(size=(5, 5))
        h = a.T @ a + np.eye(5)
        x = rng.normal(size=5)
        m, v = np.zeros(5), np.zeros(5)
        for t in range(1, 9):
            g = h @ x
            m = .9*m + .1*g
            v = .999*v + .001*g*g
            direction = (m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
            eta = .003
            y = x - eta*direction
            next_g = h @ y
            rho = (g @ next_g)/(np.linalg.norm(g)*np.linalg.norm(next_g))
            projected = 1-rho*np.linalg.norm(next_g)/np.linalg.norm(g)
            errors += [abs(projected + g@(next_g-g)/(g@g)),
                       abs(projected - eta*(g@h@direction)/(g@g))]
            checks += 1
            x = y
    # Enumerate finite-population subsets; check the exact covariance factor.
    population = rng.normal(size=(5, 3))
    mu = population.mean(axis=0)
    centered = population-mu
    covariance = centered.T @ centered/len(population)
    for b in range(1, 6):
        noise = np.array([population[list(ids)].mean(axis=0)-mu
                          for ids in itertools.combinations(range(5), b)])
        expected = (5-b)/(b*4)*covariance
        errors.append(np.max(np.abs(noise.T@noise/len(noise)-expected)))
        checks += 1
    # Full displacement and retention: direct vector sum vs explicit all pairs.
    for count in [2, 7, 19]:
        g = rng.normal(size=(count, 5))
        pair_sum = sum(2*g[u]@g[t] for t in range(count) for u in range(t))
        lhs = np.linalg.norm(g.sum(axis=0))**2
        rhs = np.sum(g*g)+pair_sum
        errors.append(abs(lhs-rhs))
        retention = np.sqrt(lhs)/np.linalg.norm(g, axis=1).sum()
        assert 0 <= retention <= 1
        checks += 1
    assert max(errors) < 1e-10
    return dict(cases=checks, max_absolute_error=float(max(errors)),
                coverage=['Adam projected curvature response on quadratics',
                          'Exact finite-population covariance',
                          'All-pair displacement and retention'])


def check_sources():
    evidence = read(HERE / 'evidence_manifest.json')
    assert evidence['generator_sha256'] == sha(HERE / '01_prepare_table.py')
    for path, digest in evidence['sources'].items():
        assert sha(ROOT / path) == digest, path
    for figure in evidence['figures']:
        assert sha(ROOT / figure['source']) == sha(ROOT / figure['target']) == figure['sha256']
    table = ROOT / evidence['table']['path']
    assert sha(table) == evidence['table']['sha256']
    table_text = table.read_text()
    assert r'\begin{tabular*}{\textwidth}' in table_text
    assert '*{13}{c}' in table_text and r'\multirow{3}' in table_text
    for row in evidence['table']['rows']:
        assert ' & '.join(row['cells']) + r' \\' in table_text
    assert sum(row['n'] for row in evidence['table']['rows']) == 129
    feedback = read(HERE / 'feedback.json')
    original = HERE.parent / '2026-11-03-adam-spike-anatomy' / feedback['source']
    assert sha(original) == feedback['source_sha256'] and len(feedback['comments']) == 12
    section = (DRAFT / 'sections/06-adam-diagnostics.tex').read_text()
    for value in ['117 of 127', '21 of 127', '82 of 129', '859', '67--69', '90--94', '38--54']:
        assert value in section, value
    for word in ['interspersed', 'illustrations', 'tab:adam-forecast', 'average precision']:
        assert word not in section
    spec = importlib.util.spec_from_file_location('retained', RETAINED / '02_build_and_verify.py')
    retained = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retained)
    return evidence, dict(retained_math=retained.check_math(),
                          additional_math=extra_math(),
                          retained_gd_sgd_evidence=retained.check_evidence(),
                          adam_claims=evidence['claims'], preserved_author_comments=12)


def main():
    evidence, record = check_sources()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tex = ['pdflatex', '--disable-installer', '-no-shell-escape', '-recorder',
           '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
           '-output-directory='+str(SCRATCH), 'bare_jrnl.tex']
    for index, command in enumerate([tex, ['bibtex', str(SCRATCH/'bare_jrnl')], tex, tex], 1):
        result = subprocess.run(command, cwd=DRAFT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH/f'build-{index}.txt').write_bytes(result.stdout)
        if result.returncode:
            print(result.stdout.decode(errors='replace')[-7000:])
            raise RuntimeError(f'Build pass {index} failed')
    log = (SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    for extra in range(3):
        if 'Rerun to get cross-references right' not in log:
            break
        result = subprocess.run(tex, cwd=DRAFT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        (SCRATCH/f'build-extra-{extra+1}.txt').write_bytes(result.stdout)
        assert result.returncode == 0
        log = (SCRATCH/'bare_jrnl.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if any(token in line.lower()
                for token in ['overfull', 'undefined', 'rerun to get', 'multiply defined'])]
    if problems:
        print('\n'.join(problems))
        raise AssertionError('Resolve layout/reference warnings before acceptance')
    expected = {str((ROOT/f['target']).resolve()) for f in evidence['figures']}
    expected |= {str((ROOT/f['target']).resolve())
                 for f in read(RETAINED/'figure_manifest.json')['figures']}
    recorded = set()
    for line in (SCRATCH/'bare_jrnl.fls').read_text().splitlines():
        if line.startswith('INPUT '):
            p = Path(line[6:])
            p = p if p.is_absolute() else DRAFT/p
            if p.suffix == '.pdf' and p.resolve().is_relative_to((DRAFT/'figures').resolve()):
                recorded.add(str(p.resolve()))
    assert expected == recorded and len(recorded) == 8
    pdf = SCRATCH/'bare_jrnl.pdf'
    with fitz.open(pdf) as document:
        text = '\n'.join(page.get_text() for page in document)
        assert '??' not in text
        record['pages'] = len(document)
    auxiliary = (SCRATCH/'bare_jrnl.aux').read_text()
    record['placements'] = {label: int(re.search(r'\\newlabel\{' + label +
        r'\}\{\{[^}]+\}\{(\d+)\}', auxiliary).group(1)) for label in [
        'sec:displacement', 'sec:gd-experiments', 'sec:adam-diagnostics', 'sec:conclusion',
        'fig:adam-overview', 'fig:adam-spikes', 'fig:adam-spike-distribution',
        'tab:adam-spike-anatomy', 'fig:adam-rho-norm-spikes',
        'appendix:loss-curvature', 'appendix:gd-raw-alignment']}
    result = subprocess.run(['pdftoppm', '-r', '100', '-png', str(pdf), str(SCRATCH/'review')],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    for target in [DRAFT/'bare_jrnl.pdf', HERE/'revised-manuscript.pdf']:
        shutil.copyfile(pdf, target)
    record.update(status='awaiting_visual_review', pdf_sha256=sha(pdf),
                  active_figures=8, verifier_sha256=sha(__file__),
                  evidence_manifest_sha256=sha(HERE/'evidence_manifest.json'),
                  build='pdflatex/BibTeX; zero overfull boxes or unresolved references',
                  source_hashes={p.relative_to(ROOT).as_posix(): sha(p) for p in [
                      DRAFT/'bare_jrnl.tex', DRAFT/'bibliography.bib',
                      *sorted((DRAFT/'sections').glob('*.tex')),
                      DRAFT/'tables/analysis036_spike_anatomy.tex']})
    (HERE/'verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['status','pages','placements','pdf_sha256',
                                           'additional_math']}, indent=2))


if __name__ == '__main__':
    main()
