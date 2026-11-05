"""Rebuild the diagnostic manuscript and render every page for human visual QA.

No training, remote execution, scientific-input mutation or new empirical fit.
Run from any directory. Outputs stay in the revision and ignored tmp/pdfs.
"""
import hashlib
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
SCRATCH = ROOT / 'tmp/pdfs/submission-synthesis-2026-11-05'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def math_checks():
    """Check the two added identities independently on nontrivial polynomials."""
    rng = np.random.default_rng(20261105)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    worst_secant = worst_loss = 0.0
    for _ in range(100):
        mat = rng.normal(size=(7, 7))
        h = (mat + mat.T) / 2  # Include indefinite objectives.
        x = rng.normal(size=7)
        q = 0.2
        loss = lambda v: 0.5 * v @ h @ v + q / 4 * np.sum(v**4)
        grad = lambda v: h @ v + q * v**3
        hessian = lambda v: h + np.diag(3 * q * v**2)
        g = grad(x)
        eta = float(rng.uniform(0.001, 0.2))
        step = -eta * g
        following = grad(x + step)
        measured = (1 - g @ following / (g @ g)) / eta
        secant = step @ (following - g) / (step @ step)
        worst_secant = max(worst_secant, abs(measured - secant))
        np.testing.assert_allclose(measured, secant, rtol=1e-10, atol=1e-10)
        # Applied direction need not be proportional to the raw gradient.
        increment = rng.normal(size=7) * 0.15
        remainder = sum(
            w / 2 * (1 - (a + 1) / 2)
            * (increment @ hessian(x + (a + 1) / 2 * increment) @ increment)
            for a, w in zip(nodes, weights)
        )
        observed = loss(x + increment) - loss(x)
        predicted = g @ increment + remainder
        worst_loss = max(worst_loss, abs(observed - predicted))
        np.testing.assert_allclose(observed, predicted, rtol=1e-10, atol=1e-10)
    return dict(cases=100, inverse_bb1_max_error=worst_secant,
                finite_step_integral_max_error=worst_loss)


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((HERE / 'baseline_manifest.json').read_text())
    protected = {
        p: digest for p, digest in baseline['source_sha256'].items()
        if '/figures/' in p or '/previous-submission/' in p or '/tables/' in p
    }
    for path, digest in protected.items():
        assert sha(ROOT / path) == digest, path
    evidence = json.loads((HERE / 'evidence_selection.json').read_text())
    for item in evidence['source_results'] + evidence['protocol_sources']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    subprocess.run(["python", str(HERE / "verify_implications.py")],
                   check=True, capture_output=True, timeout=120)
    checks = math_checks()
    command = ['pdflatex', '--disable-installer', '-no-shell-escape', '-recorder',
               '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-output-directory=' + str(SCRATCH), 'bare_jrnl.tex']
    for i, cmd in enumerate([command, ['bibtex', str(SCRATCH / 'bare_jrnl')],
                             command, command, command], 1):
        proc = subprocess.run(cmd, cwd=DRAFT, capture_output=True, timeout=120)
        (SCRATCH / f'build-{i}.txt').write_bytes(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f'Build {i} failed; see {SCRATCH}')
    log = (SCRATCH / 'bare_jrnl.log').read_text(errors='replace')
    issues = [line for line in log.splitlines() if any(
        word in line.lower() for word in
        ['overfull', 'undefined', 'rerun to get', 'multiply defined'])]
    pdf = SCRATCH / 'bare_jrnl.pdf'
    with fitz.open(pdf) as doc:
        text = '\n'.join(page.get_text() for page in doc)
        assert '??' not in text
        page_content = [dict(page=i + 1, characters=len(page.get_text()))
                        for i, page in enumerate(doc)]
        (SCRATCH / 'manuscript-text.txt').write_text(text, encoding='utf-8')
        page_count = len(doc)
    subprocess.run(['pdftoppm', '-r', '110', '-png', str(pdf),
                    str(SCRATCH / 'review')], check=True, capture_output=True,
                   timeout=120)
    active_inputs = re.findall(r'^\\input\{([^}]+)\}',
                              (DRAFT / 'bare_jrnl.tex').read_text(), re.M)
    report = dict(status='needs_layout_fix' if issues else 'pending_visual_review',
                  pages=page_count, issues=issues, pdf_sha256=sha(pdf),
                  page_content=page_content, mathematical_checks=checks,
                  protected_files_verified=len(protected),
                  evidence_sources_verified=len(evidence['source_results'])
                  + len(evidence['protocol_sources']), active_inputs=active_inputs,
                  sources={p.relative_to(ROOT).as_posix(): sha(p) for p in
                           [DRAFT / 'bare_jrnl.tex', DRAFT / 'bibliography.bib',
                            *sorted((DRAFT / 'sections').glob('*.tex')),
                            DRAFT / 'tables/submission_interventions.tex']})
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n',
                                          encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'pages', 'issues',
                                           'mathematical_checks']}))
    if issues:
        raise RuntimeError('Resolve build warnings before delivery.')
    shutil.copyfile(pdf, HERE / 'revised-manuscript.pdf')
    shutil.copyfile(pdf, DRAFT / 'bare_jrnl.pdf')


if __name__ == '__main__':
    main()
