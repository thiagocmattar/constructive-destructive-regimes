"""Verify the IEEE revision's data preservation, assets and compiled manuscript."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = '198a969'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def table_rows(name):
    text = (ROOT / 'manuscript/draft/tables' / name).read_text()
    rows = [line for line in text.splitlines() if line.startswith(('MLP &', 'ResNet-32 &', 'Transformer &'))]
    assert len(rows) == 9, name
    parsed = {}
    for row in rows:
        cells = [s.strip() for s in row.removesuffix(r'\\').split('&')]
        task = {'MLP': 'mlp', 'ResNet-32': 'resnet', 'Transformer': 'transformer'}[cells[0]]
        key = (task, float(cells[1]))
        assert key not in parsed
        parsed[key] = cells[2:]
    return parsed


def check_number(cell, value):
    scientific = re.search(r'\$(-?[0-9.]+).*10\^\{(-?\d+)\}', cell)
    if scientific:
        mantissa, exponent = scientific.groups()
        unit = 10 ** (int(exponent) - len(mantissa.split('.')[1]))
        displayed = float(mantissa) * 10 ** int(exponent)
    else:
        displayed, unit = float(cell), 1e-4
    assert abs(displayed - value) <= .5001 * unit, (cell, value)


def main():
    manifest_path = HERE / 'single_rows_figure_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'passed' and len(manifest['figures']) == 8
    assert manifest['script_sha256'] == sha(HERE / '07_plot_single_rows.py')
    for key in ['arrays', 'results']:
        assert sha(ROOT / manifest[key]) == manifest[key + '_sha256']
    baseline = json.loads(subprocess.check_output([
        'git', 'show', BASELINE + ':' + manifest_path.relative_to(ROOT).as_posix()], cwd=ROOT))
    current = {c['tag']: c for f in manifest['figures'] for c in f['curves']}
    arrays = np.load(ROOT / manifest['arrays'])
    same_y, shifted_x, same_x = [], [], []
    for figure in baseline['figures']:
        name = figure['name']
        for old in figure['curves']:
            new = current[old['tag']]
            assert old['y_sha256'] == new['y_sha256'] == array_sha(arrays[old['tag'] + '_y'])
            same_y.append(old['tag'])
            if name in ['loss_log', 'normalized_displacement']:
                assert old['x_sha256'] == new['x_sha256']
                same_x.append(old['tag'])
            else:
                old_x = (np.arange(1, 3000) if name == 'rho' else
                         np.arange(49, 3000) if name == 'gradient_norms_smoothed' else
                         np.arange(50, 3000))
                assert old['x_sha256'] == array_sha(old_x)
                np.testing.assert_array_equal(arrays[old['tag'] + '_x'], old_x + 1)
                shifted_x.append(old['tag'])
    assert len(same_y) == 54 and len(shifted_x) == 36 and len(same_x) == 18

    graphics = []
    for figure in manifest['figures']:
        for key in ['pdf', 'render']:
            assert sha(ROOT / figure[key]) == figure[key + '_sha256']
        doc = fitz.open(ROOT / figure['pdf'])
        assert len(doc) == 1 and abs(doc[0].rect.width / 72 - 7.16) < .001
        text = doc[0].get_text()
        assert all(word not in text for word in ['Seed 0', 'Unsmoothed', 'Trailing'])
        assert all(f'({chr(97 + i)})' in text for i in range(3 * figure['rows']))
        fonts = []
        for entry in doc[0].get_fonts(full=True):
            assert entry[2] != 'Type3'
            name, extension, kind, content = doc.extract_font(entry[0])
            assert content, 'Every graphics font must be embedded.'
            fonts.append(dict(name=name, extension=extension, kind=kind, bytes=len(content)))
        graphics.append(dict(name=figure['name'], panels=3 * figure['rows'],
                             size_inches=[doc[0].rect.width / 72, doc[0].rect.height / 72], fonts=fonts))

    assets = json.loads((HERE / 'manuscript_assets.json').read_text())
    assert assets['status'] == 'passed'
    assert assets['script_sha256'] == sha(HERE / '08_build_manuscript_assets.py')
    assert assets['source_figure_manifest_sha256'] == sha(manifest_path)
    assert assets['source_results_sha256'] == sha(HERE / 'results.json')
    assert assets['source_review_audit_sha256'] == sha(HERE / 'section_iv_review_checks.json')
    assert assets['source_inputs_sha256'] == sha(ROOT / 'cache/eos-multitask/inputs.json')
    assert len(assets['outputs']) == 8
    for output in assets['outputs']:
        assert sha(ROOT / output['path']) == output['sha256']
        if output['path'].endswith('.pdf'):
            assert sha(ROOT / output['path']) == sha(ROOT / output['source'])

    paper_path = ROOT / 'manuscript/artifacts/draft/bare_jrnl.pdf'
    paper = fitz.open(paper_path)
    texts = [re.sub(r'\s+', ' ', page.get_text()) for page in paper]
    section_iv = next(i + 1 for i, t in enumerate(texts) if 'IV. EMPIRICAL CHARACTERIZATION' in t)
    section_v = next(i + 1 for i, t in enumerate(texts) if 'V. MECHANISMS OF TRAJECTORY' in t)
    figure_pages = {str(n): next(i + 1 for i, t in enumerate(texts) if f'Fig. {n}:' in t)
                    for n in [1, 2, 3, 8]}
    assert all(section_iv <= figure_pages[str(n)] < section_v for n in [1, 2, 3])
    assert any('TABLE I:' in t for t in texts)
    assert any('APPENDIX D' in t for t in texts)
    assert all(any(f'TABLE {n}:' in t for t in texts) for n in ['I', 'II', 'III', 'IV'])
    assert any('44.2%' in t for t in texts) and any('43.7%' in t for t in texts)
    assert all('??' not in t for t in texts)
    log = (paper_path.parent / 'bare_jrnl.log').read_text(errors='replace')
    forbidden = ['Overfull', 'undefined references', 'undefined citations',
                 'multiply defined', 'Rerun to get cross-references right']
    assert all(word not in log for word in forbidden)
    # Parse all four tables independently and reconstruct the new quantities
    # from source metrics, using the gradient-dot loss form rather than q.
    source = json.loads((HERE / 'results.json').read_text())
    audit = json.loads((HERE / 'section_iv_review_checks.json').read_text())
    assert audit['script_sha256'] == sha(HERE / '10_review_section_iv.py')
    audit_by_id = {r['run_id']: r for r in audit['runs']}
    endpoints = table_rows('analysis024_endpoints.tex')
    temporal = table_rows('analysis024_temporal_accounting.tex')
    loss_audit = table_rows('analysis024_loss_audit.tex')
    spectra = table_rows('analysis024_spectral_context.tex')
    expected_keys = {(r['task'], r['lr']) for r in source['runs']}
    assert all(set(t) == expected_keys for t in [endpoints, temporal, loss_audit, spectra])
    assert all(len(cells) == width for table, width in
               [(endpoints, 5), (temporal, 4), (loss_audit, 5), (spectra, 2)] for cells in table.values())
    loss_pairs = 0
    for r in source['runs']:
        key, a = (r['task'], r['lr']), audit_by_id[r['run_id']]
        path = ROOT / a['source']
        assert sha(path) == a['source_sha256'] == r['source_sha256']['metrics.jsonl']
        rows = [json.loads(s) for s in path.read_text().splitlines()][:3000]
        assert [v['update'] for v in rows] == list(range(1, 3001))
        raw = [v['streams']['raw'] for v in rows]
        norm = np.array([v['norm'] for v in raw])
        cosine = np.array([v['cosine'] for v in raw[1:]])
        dot = cosine * norm[:-1] * norm[1:]
        lag_one = 2 * dot.sum()
        total = raw[-1]['B']**2 - np.square(norm).sum()
        distance, length = r['lr'] * raw[-1]['B'], r['lr'] * norm.sum()
        for cell, value in zip(endpoints[key], [r['final_loss'], r['late_1000_cosine_mean'],
                r['final_B_over_A'], r['late_1000_q_mean'], distance]):
            check_number(cell, value)
        for cell, value in zip(temporal[key], [length, lag_one, total-lag_one, total]):
            check_number(cell, value)
        loss = np.array([v.get('train_loss', v.get('batch_objective_estimate')) for v in rows])
        actual = np.diff(loss)
        surrogate = -r['lr'] * (np.square(norm[:-1]) + dot) / 2
        residual = np.median(np.abs(actual-surrogate) / (r['lr'] * np.square(norm[:-1])))
        check_number(loss_audit[key][1], residual)
        for i, threshold in enumerate([0, 1e-7, 1e-6, 1e-5]):
            floor = threshold * np.maximum(np.abs(loss[:-1]), np.abs(loss[1:]))
            keep = (np.abs(actual) > floor) & (np.abs(surrogate) > floor)
            count = np.count_nonzero((actual[keep] > 0) != (surrogate[keep] > 0))
            cell = loss_audit[key][0 if i == 0 else i+1]
            assert cell == f'{count}/{int(keep.sum())}', (key, threshold, cell)
        check_number(spectra[key][1], r['final_sampled_eta_lambda'])
        assert ('Full' in spectra[key][0]) == (r['task'] == 'mlp')
        loss_pairs += len(actual)
    assert loss_pairs == 26991
    # Section III equations and the original Appendix D raw-cosine block stay intact.
    methods_path = 'manuscript/draft/sections/03-methods.tex'
    old_methods = subprocess.check_output(['git', 'show', 'b5a5655:' + methods_path], cwd=ROOT).decode()
    new_methods = (ROOT / methods_path).read_text()
    equations = r'\\begin\{equation\}(.*?)\\end\{equation\}'
    assert re.findall(equations, old_methods, re.S) == re.findall(equations, new_methods, re.S)
    appendix_path = 'manuscript/draft/sections/07-gd-diagnostics.tex'
    old_appendix = subprocess.check_output(['git', 'show', 'b5a5655:' + appendix_path], cwd=ROOT).decode()
    raw_block = old_appendix[old_appendix.index(r'Figure~\ref{fig:gd-raw-alignment}'):].strip()
    assert raw_block in (ROOT / appendix_path).read_text()
    main_source = (ROOT / 'manuscript/draft/sections/04-gd-experiments.tex').read_text()
    assert 'final all-pair interaction' in main_source and '1,325' in main_source
    assert 'Hessian eigenvalue' not in main_source
    assert 'pytorchWordLanguageModelPinned' in (ROOT / appendix_path).read_text()
    bibliography = (ROOT / 'manuscript/draft/bibliography.bib').read_text()
    assert '13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15/word_language_model' in bibliography
    record = dict(status='passed', baseline_commit=BASELINE, unchanged_y_curves=len(same_y),
                  x_shifted_by_one=len(shifted_x), x_unchanged=len(same_x), graphics=graphics,
                  manuscript=dict(path=paper_path.relative_to(ROOT).as_posix(), sha256=sha(paper_path),
                                  pages=len(paper), section_iv_page=section_iv, section_v_page=section_v,
                                  figure_pages=figure_pages, overfull_boxes=0, unresolved_references=0),
                  endpoint_rows_checked=9, total_table_rows_checked=36, loss_pairs_checked=loss_pairs,
                  table_verification='Independent parsing, metric-source hashes and gradient-dot loss reconstruction',
                  section_iii_equations_unchanged=True, raw_alignment_block_unchanged=True,
                  manuscript_source_sha256={name: sha(ROOT / 'manuscript/draft' / name) for name in [
                      'bare_jrnl.tex', 'bibliography.bib', 'sections/03-methods.tex',
                      'sections/04-gd-experiments.tex', 'sections/07-gd-diagnostics.tex']},
                  manuscript_assets_sha256=sha(HERE / 'manuscript_assets.json'),
                  script_sha256=sha(__file__))
    (HERE / 'manuscript_verification.json').write_text(json.dumps(record, indent=2) + '\n',
                                                   encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'passed', 'unchanged_y_curves': len(same_y),
                      'pages': len(paper), 'main_figure_pages': figure_pages, 'embedded_graphics': len(graphics)}))


if __name__ == '__main__':
    main()
