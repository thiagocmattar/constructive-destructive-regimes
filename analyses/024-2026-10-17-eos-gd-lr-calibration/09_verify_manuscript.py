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
    assert assets['source_figure_manifest_sha256'] == sha(manifest_path)
    assert assets['source_results_sha256'] == sha(HERE / 'results.json')
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
    assert all('??' not in t for t in texts)
    log = (paper_path.parent / 'bare_jrnl.log').read_text(errors='replace')
    forbidden = ['Overfull', 'undefined references', 'undefined citations',
                 'multiply defined', 'Rerun to get cross-references right']
    assert all(word not in log for word in forbidden)
    # A generated table must include all nine independently verified endpoints.
    source = json.loads((HERE / 'results.json').read_text())
    table = (ROOT / 'manuscript/draft/tables/analysis024_endpoints.tex').read_text()
    body_rows = [line for line in table.splitlines() if line.startswith(('MLP &', 'ResNet-32 &', 'Transformer &'))]
    assert len(body_rows) == 9
    for line in body_rows:
        # Source order differs from the table; identify by model and LR instead.
        cells = [s.strip() for s in line.split('&')]
        task = {'MLP': 'mlp', 'ResNet-32': 'resnet', 'Transformer': 'transformer'}[cells[0]]
        match = next(x for x in source['runs'] if x['task'] == task and x['lr'] == float(cells[1]))
        assert abs(float(cells[2]) - match['final_loss']) <= .00005001
        assert abs(float(cells[3]) - match['late_1000_cosine_mean']) <= .00005001
        assert abs(float(cells[4]) - match['final_B_over_A']) <= .00005001
        scientific = re.search(r'\$([0-9.]+).*10\^\{(-?\d+)\}', cells[5])
        if scientific:
            power = int(scientific.group(2))
            q = float(scientific.group(1)) * 10 ** power
            tolerance = .005001 * 10 ** power
        else:
            q, tolerance = float(cells[5]), .00005001
        assert abs(q - match['late_1000_q_mean']) <= tolerance
        assert abs(float(cells[6].removesuffix(r'\\').strip()) - match['final_sampled_eta_lambda']) <= .00005001
    record = dict(status='passed', baseline_commit=BASELINE, unchanged_y_curves=len(same_y),
                  x_shifted_by_one=len(shifted_x), x_unchanged=len(same_x), graphics=graphics,
                  manuscript=dict(path=paper_path.relative_to(ROOT).as_posix(), sha256=sha(paper_path),
                                  pages=len(paper), section_iv_page=section_iv, section_v_page=section_v,
                                  figure_pages=figure_pages, overfull_boxes=0, unresolved_references=0),
                  endpoint_rows_checked=9, manuscript_assets_sha256=sha(HERE / 'manuscript_assets.json'),
                  script_sha256=sha(__file__))
    (HERE / 'manuscript_verification.json').write_text(json.dumps(record, indent=2) + '\n',
                                                   encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'passed', 'unchanged_y_curves': len(same_y),
                      'pages': len(paper), 'main_figure_pages': figure_pages, 'embedded_graphics': len(graphics)}))


if __name__ == '__main__':
    main()
