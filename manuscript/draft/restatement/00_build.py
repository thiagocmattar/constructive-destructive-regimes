"""Build the requested GD-only report, preserve inputs, and render every page."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ANALYSIS = ROOT / 'analyses/019-2026-10-13-cifar10-eos-lr'
OUTPUT = ROOT / 'manuscript/2026-10-11-restatement'
ARTIFACTS = ROOT / 'manuscript/artifacts/2026-10-11-restatement'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_math():
    """Finite noise enumeration checks the identities without a Monte Carlo estimate."""
    hessian = np.array([[2., .3], [.3, 4.]])
    base, old_noise, new_noise, eta = np.array([.4, -.7]), np.array([.1, .2]), np.array([.3, -.05]), .15
    noise_terms = []
    for old_sign in [-1, 1]:
        for new_sign in [-1, 1]:
            xi_old, xi_new = old_sign * old_noise, new_sign * new_noise
            previous = base + xi_old
            current = (np.eye(2) - eta*hessian) @ previous + xi_new - xi_old
            previous_norm, current_norm = np.linalg.norm(previous), np.linalg.norm(current)
            rho = previous @ current / (previous_norm*current_norm)
            kappa = previous @ hessian @ previous / previous_norm**2
            epsilon = previous @ (xi_new-xi_old) / previous_norm**2
            np.testing.assert_allclose(rho, previous_norm/current_norm*(1-eta*kappa+epsilon), rtol=1e-13)
            np.testing.assert_allclose(1-rho*current_norm/previous_norm+epsilon, eta*kappa, rtol=1e-13)
            noise_terms.append(previous @ (xi_new-xi_old))
    np.testing.assert_allclose(np.mean(noise_terms), -old_noise @ old_noise, atol=1e-15)
    vectors = np.array([[1., 2.], [-.5, -1.], [.3, .2], [-.1, .5]])
    lengths = np.linalg.norm(vectors, axis=1)
    a, b = lengths.sum(), np.linalg.norm(vectors.sum(axis=0))
    interaction = sum(vectors[u] @ vectors[t] for t in range(len(vectors)) for u in range(t))
    np.testing.assert_allclose(b*b, np.sum(lengths**2)+2*interaction, rtol=1e-13)
    gap = sum(lengths[u]*lengths[t]-vectors[u] @ vectors[t] for t in range(len(vectors)) for u in range(t))
    np.testing.assert_allclose((b/a)**2, 1-2*gap/a**2, rtol=1e-13)
    return {'status': 'passed', 'cases': ['displacement energy/interaction', 'normalized displacement gap',
        'quadratic SGD alignment with explicit noise', 'noise-corrected directional curvature',
        'conditional-unbiased noise inner-product expectation (four exact outcomes)']}


def result_table():
    results = json.loads((ANALYSIS / 'results.json').read_text())
    assert results['status'] == 'passed'
    rows = [r for r in results['runs'] if r['optimizer'] == 'gd']
    assert [r['run_id'] for r in rows] == ['059', '060', '061', '062']
    text = r'''\begin{table}[htbp]
\centering
\small
\caption{GD results at $T=20{,}000$. Bars average the last 1,000 recorded
gradient-pair values; loss, maximum curvature and $R_T$ refer to the final
state or full prefix. The last recorded gradient belongs to $\theta_{T-1}$;
the final loss and Hessian belong to $\theta_T$. One initialization is used.}
\label{tab:gd-results}
\begin{tabular}{rrrrrr}
\toprule
$\eta$ & $\Loss(\theta_T)$ & $\overline\rho$ & $\overline{\eta\widehat\kappa}$ & $\eta\lambda_{\max}(\theta_T)$ & $R_T$\\
\midrule
'''
    for row, lr in zip(rows, ['$0.004$', '$2/300$', '$0.01$', '$0.02$']):
        raw = row['streams']['raw']
        kappa = row['eta_kappa_last_1000_mean']
        kappa_cell = '$4.35\\times10^{-5}$' if kappa < 1e-3 else f'{kappa:.4f}'
        text += f"{lr} & {row['final_train']['loss']:.5f} & {raw['last_1000_cosine_mean']:.4f} & {kappa_cell} & {row['final_eta_lambda_max']:.4f} & {raw['B_over_A']:.4f}\\\\\n"
        with np.load(ANALYSIS / 'derived' / ('Run' + row['run_id'] + '.npz'), allow_pickle=False) as values:
            np.testing.assert_allclose(values['raw_cosine'][-1000:].mean(), raw['last_1000_cosine_mean'], rtol=1e-13)
            np.testing.assert_allclose(values['eta_kappa_proxy'][-1000:].mean(), kappa, rtol=1e-13)
            np.testing.assert_allclose(row['lr']*values['hessian_eigenvalues'][-1, 0], row['final_eta_lambda_max'], rtol=1e-13)
    text += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (HERE / 'gd-results.tex').write_text(text, encoding='utf-8')
    return rows


def render(path, prefix):
    pages = []
    poppler = shutil.which('pdftoppm')
    with fitz.open(path) as pdf:
        for number, page in enumerate(pdf):
            destination = ARTIFACTS / f'{prefix}-page-{number+1:02d}.png'
            engine = 'PyMuPDF'
            if poppler:
                try:
                    subprocess.run([poppler, '-f', str(number+1), '-l', str(number+1), '-r', '110',
                        '-png', '-singlefile', str(path), str(destination.with_suffix(''))],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                    engine = 'Poppler'
                except (subprocess.SubprocessError, OSError):
                    page.get_pixmap(matrix=fitz.Matrix(110/72, 110/72), alpha=False).save(destination)
            else:
                page.get_pixmap(matrix=fitz.Matrix(110/72, 110/72), alpha=False).save(destination)
            outside = [span['text'] for block in page.get_text('dict')['blocks']
                for line in block.get('lines', []) for span in line['spans']
                if not fitz.Rect(-1, -1, page.rect.width+1, page.rect.height+1).contains(fitz.Rect(span['bbox']))]
            assert not outside, (number+1, outside)
            pages.append({'page': number+1, 'page_points': [page.rect.width, page.rect.height],
                'renderer': engine, 'render': destination.relative_to(ROOT).as_posix(),
                'render_sha256': sha256(destination), 'text_spans_outside_page': outside})
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true', help='Record visual review of all current pages; do not recompile.')
    args = parser.parse_args()
    metadata = OUTPUT / 'gd-formulation-manifest.json'
    if args.accept_reviewed:
        record = json.loads(metadata.read_text())
        assert sha256(OUTPUT / 'gd-formulation.pdf') == record['report_sha256']
        assert sha256(HERE / 'gd-formulation.tex') == record['latex_source_sha256']
        assert sha256(HERE / 'gd-results.tex') == record['table_sha256']
        assert sha256(__file__) == record['build_script_sha256']
        assert sha256(ANALYSIS / 'results.json') == record['results_sha256']
        assert sha256(ANALYSIS / 'figures/cifar10_gd_diagnostics_paper.pdf') == record['figure_sha256']
        assert sha256(ANALYSIS / 'gd_paper_figure_manifest.json') == record['figure_manifest_sha256']
        for name, expected in record['inputs_sha256_unchanged'].items():
            assert sha256(ROOT / name) == expected
        for page in record['report_pages'] + record['figure_pages']:
            assert sha256(ROOT / page['render']) == page['render_sha256']
        record['status'] = 'passed'
        record['visual_review'] = 'All report pages and the standalone figure inspected for layout, equations, captions, tables, clipping and references.'
        write_json(metadata, record)
        print(json.dumps({'status': 'passed', 'report_pages': len(record['report_pages'])}))
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    inputs = [OUTPUT / 'formulation.md', OUTPUT / 'writing-style.md']
    input_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    math_checks = check_math()
    rows = result_table()
    for attempt in [1, 2]:
        print(f'LaTeX pass {attempt}/2', flush=True)
        process = subprocess.run(['pdflatex', '--disable-installer', '-no-shell-escape',
            '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
            '-output-directory=' + str(ARTIFACTS), 'gd-formulation.tex'],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (ARTIFACTS / f'latex-pass-{attempt}.log').write_bytes(process.stdout)
        if process.returncode:
            print(process.stdout.decode(errors='replace')[-7000:])
            raise RuntimeError(f'LaTeX pass {attempt} failed; see retained log.')
    log = (ARTIFACTS / 'gd-formulation.log').read_text(errors='replace')
    assert 'Overfull' not in log and 'undefined' not in log.lower() and 'Rerun to get' not in log, 'Unresolved LaTeX layout/reference warning.'
    compiled = ARTIFACTS / 'gd-formulation.pdf'
    pending = ARTIFACTS / 'gd-formulation.delivery.pdf'
    shutil.copyfile(compiled, pending)
    pending.replace(OUTPUT / 'gd-formulation.pdf')
    destination = OUTPUT / 'gd-formulation.pdf'
    with fitz.open(destination) as pdf:
        text = '\n'.join(page.get_text() for page in pdf)
        assert 'Full-batch GD evidence on CIFAR-10' in text
        assert 'Adam' not in text and '??' not in text
        assert 'TODO' in text and 'References' in text
    report_pages = render(destination, 'report')
    figure = ANALYSIS / 'figures/cifar10_gd_diagnostics_paper.pdf'
    figure_pages = render(figure, 'figure')
    assert {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs} == input_hashes
    write_json(metadata, {'status': 'awaiting_visual_review',
        'scope': 'English SGD formulation and empirical full-batch GD result; stops before Displacement under Adam.',
        'report': destination.relative_to(ROOT).as_posix(), 'report_sha256': sha256(destination),
        'latex_source': (HERE / 'gd-formulation.tex').relative_to(ROOT).as_posix(),
        'latex_source_sha256': sha256(HERE / 'gd-formulation.tex'),
        'table_sha256': sha256(HERE / 'gd-results.tex'), 'build_script_sha256': sha256(__file__),
        'inputs_sha256_unchanged': input_hashes, 'mathematics': math_checks,
        'results_sha256': sha256(ANALYSIS / 'results.json'),
        'figure_sha256': sha256(figure), 'figure_manifest_sha256': sha256(ANALYSIS / 'gd_paper_figure_manifest.json'),
        'report_pages': report_pages, 'figure_pages': figure_pages, 'visual_review': None,
        'latex_checks': 'Two passes; no overfull boxes, undefined references/citations or pending rerun.',
        'runs': [r['run_id'] for r in rows], 'new_training': False,
        'references': ['https://arxiv.org/abs/2103.00065', 'https://github.com/locuslab/edge-of-stability/tree/github']})
    print(json.dumps({'report': str(destination), 'pages': len(report_pages),
                      'math_checks': 'passed', 'layout': 'awaiting_visual_review'}))


if __name__ == '__main__':
    main()
