"""Audit the focused manuscript revision against retained repetition-level results."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np
from pypdf import PdfReader

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
DRAFT = ROOT / 'manuscript/draft'
BUILD = ROOT / 'manuscript/artifacts/draft'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    verified = read(HERE / 'verification.json')
    assert digest(HERE / 'results.json') == verified['results_sha256']
    results = read(HERE / 'results.json')
    manifest = read(HERE / 'paper_figure_manifest.json')
    reduced = read(HERE / 'reduction.json')
    assert manifest['source_results_sha256'] == digest(HERE / 'results.json')
    assert manifest['source_reduction_sha256'] == digest(HERE / 'reduction.json')
    blocks_checked = 0
    for curve in manifest['curves']:
        stage = 'frozen' if curve['figure'] == 'batch_size_fixed_parameters' else 'moving'
        blocks = []
        for seed in range(3):
            values = []
            for b in curve['batches']:
                cells = [c for c in results[stage] if (c['seed'], c['checkpoint'], c['batch'])
                         == (seed, curve['checkpoint'], b)]
                assert len(cells) == (32 if stage == 'frozen' else 1 if b == 5000 else 8)
                if curve['field'] == 'predicted_rho':
                    c = cells[0]
                    value = c['signal_energy'] / (c['signal_energy'] + c['predicted_variance'])
                else:
                    value = np.mean([c[curve['field']] for c in cells])
                values.append(value)
            blocks.append(values)
        blocks = np.array(blocks)
        np.testing.assert_allclose(curve['initialization_means'], blocks, atol=1e-14, rtol=0)
        for name, expected in [('mean', blocks.mean(0)), ('minimum', blocks.min(0)), ('maximum', blocks.max(0))]:
            np.testing.assert_allclose(curve[name], expected, atol=1e-14, rtol=0)
        blocks_checked += blocks.size
    assert len(manifest['curves']) == 18 and blocks_checked == 270

    section = (DRAFT / 'sections/05-sgd-noise-experiments.tex').read_text()
    forbidden = [r'\mu_k', r'\chi_b', r'q_k', r'R_K', r'\rho_k', 'HVP', 'fork', r'\begin{equation}', r'\begin{table}']
    for term in forbidden:
        assert term not in section, term
    assert section.count(r'\includegraphics') == 2
    assert '08-sgd-noise-diagnostics' not in (DRAFT / 'bare_jrnl.tex').read_text()

    def mean(stage, checkpoint, b, field, seed=None):
        if seed is None:
            return float(np.mean([mean(stage, checkpoint, b, field, s) for s in range(3)]))
        return float(np.mean([c[field] for c in results[stage]
                              if (c['seed'], c['checkpoint'], c['batch']) == (seed, checkpoint, b)]))

    statements = []
    for stage, cp, b, field, digits, signed, absolute in [
        ('frozen', 0, 2048, 'mean_rho', 3, False, False),
        ('frozen', 1000, 2048, 'mean_rho', 3, False, False),
        ('moving', 0, 32, 'mean_rho', 3, False, False),
        ('moving', 0, 32, 'mean_full_rho', 3, True, False),
        ('moving', 0, 5000, 'mean_rho', 3, False, False),
        ('moving', 2000, 5000, 'mean_rho', 3, False, False),
        ('moving', 2000, 32, 'mean_rho', 3, False, False),
        ('moving', 2000, 5000, 'R', 4, False, False),
        ('moving', 2000, 32, 'R', 4, False, False),
        ('moving', 2000, 5000, 'loss_change', 4, False, True),
        ('moving', 2000, 32, 'loss_change', 4, False, False),
    ]:
        value = mean(stage, cp, b, field)
        rendered = format(abs(value) if absolute else value, ('+' if signed else '') + f'.{digits}f')
        assert rendered in section, (field, rendered)
        statements.append(dict(stage=stage, checkpoint=cp, batch=b, field=field, value=value, displayed=rendered))
    for seed in range(3):
        assert mean('moving', 2000, 32, 'loss_change', seed) > 0
        assert mean('moving', 2000, 2048, 'final_loss', seed) < mean('moving', 2000, 5000, 'final_loss', seed)
        assert mean('moving', 0, 32, 'loss_change', seed) < 0
    cells = [c for c in reduced['frozen'] if c['batch'] != 5000]
    variance_error = max(abs(c['mean_noise_energy'] / c['predicted_variance'] - 1) for c in cells)
    projection_error = max(abs(c['mean_noise_projection'] / c['predicted_variance'] + 1) for c in cells)
    cosine_error = max(abs(c['mean_rho'] - c['predicted_rho']) for c in cells)
    outside = sum(not (c['mean_rho_conditional_ci95'][0] <= c['predicted_rho'] <= c['mean_rho_conditional_ci95'][1]) for c in cells)
    # The paper gives conservative two-decimal percentage bounds ('within').
    assert np.ceil(10000 * variance_error) / 100 == .50
    assert np.ceil(10000 * projection_error) / 100 == .74
    assert f'{cosine_error:.4f}' == '0.0075' and outside == 21 and len(cells) == 36
    for value in ['0.50', '0.74', '0.0075', '21 of the 36']:
        assert value in section

    fonts = {}
    for name, asset in manifest['pdfs'].items():
        source, target = ROOT / asset['source'], ROOT / asset['target']
        assert digest(source) == digest(target) == asset['sha256']
        pdf = PdfReader(source)
        assert len(pdf.pages) == 1 and abs(float(pdf.pages[0].mediabox.width) - 7.16 * 72) < .001
        rows = subprocess.check_output(['pdffonts', str(source)], stderr=subprocess.DEVNULL).decode().splitlines()[2:]
        assert rows and all('Type 3' not in row for row in rows)
        assert all(re.search(r'\byes\s+yes\s+yes\b', row) for row in rows)
        fonts[name] = dict(embedded=True, type3=False, records=len(rows))

    before = subprocess.check_output(['git', 'show', '29519c7:manuscript/draft/sections/03-methods.tex'], cwd=ROOT)
    assert before == (DRAFT / 'sections/03-methods.tex').read_bytes()
    old_legacy = subprocess.check_output(['git', 'show', '29519c7:manuscript/draft/sections/04-results.tex'], cwd=ROOT).decode()
    pattern = r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}'
    assert re.findall(pattern, old_legacy, re.S) == re.findall(pattern, (DRAFT / 'sections/04-results.tex').read_text(), re.S)
    log = (BUILD / 'bare_jrnl.log').read_text(errors='replace')
    for term in ['Overfull \\hbox', 'Overfull \\vbox', 'undefined references', 'undefined citations', 'multiply defined', '! LaTeX Error']:
        assert term not in log, term
    aux = (BUILD / 'bare_jrnl.aux').read_text()
    assert r'\newlabel{eq:noise-cosine-approx}{{19}' in aux
    pages = PdfReader(BUILD / 'bare_jrnl.pdf').pages
    texts = [page.extract_text() for page in pages]
    compact = [re.sub(r'[^A-Z]', '', t.upper()) for t in texts]
    assert not any('APPENDIXE' in t for t in compact)
    figure_page = next(i + 1 for i, t in enumerate(texts) if 'Fig. 4:' in t and 'Fig. 5:' in t)
    next_section_page = next(i + 1 for i, t in enumerate(compact) if 'VIMECHANISMSOFTRAJECTORYSTABILITY' in t)
    assert figure_page < next_section_page
    report = dict(status='passed',script_sha256=digest(Path(__file__)),initialization_values_checked=blocks_checked,
                  plotted_means_checked=90,plotted_minima_maxima_checked=180,
                  numerical_statements=statements,variance_max_relative_error=variance_error,
                  projection_max_relative_error=projection_error,cosine_max_absolute_error=cosine_error,
                  cosine_intervals_excluding_prediction=outside,stochastic_cells=36,fonts=fonts,
                  section_iii_unchanged=True,legacy_equations_unchanged=True,
                  pages=len(pages),both_new_figures_page=figure_page,next_section_page=next_section_page,
                  overfull_boxes=0,unresolved_references=0,pdf_sha256=digest(BUILD/'bare_jrnl.pdf'),
                  figure_manifest_sha256=digest(HERE/'paper_figure_manifest.json'),
                  sources={name:digest(DRAFT/name) for name in ['bare_jrnl.tex','sections/01-introduction.tex',
                           'sections/03-methods.tex','sections/04-results.tex','sections/05-sgd-noise-experiments.tex']})
    (HERE / 'paper_revision_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ['status','initialization_values_checked','plotted_means_checked','pages','both_new_figures_page','next_section_page']}))


if __name__ == '__main__':
    main()
