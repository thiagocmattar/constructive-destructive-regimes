"""Check fresh-fit values, unchanged panels, source identities and PDF layout."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'figure_revision_20261114'
NAME = '01-three-model-temporal-structure.pdf'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main(visual):
    manifest = read(HERE / 'figure_manifest.json')
    assert manifest['generator_sha256'] == sha(HERE / '01_figure.py')
    assert manifest['style_generator_sha256'] == sha(OLD / '01_figures.py')
    for path, digest in manifest['sources'].items(): assert sha(ROOT / path) == digest, path
    result = read(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_results.json')
    for path, digest in result['sources'].items(): assert sha(ROOT / path) == digest, path
    assert result['completed_steps'] == list(range(390,451))
    assert result['gradient_states'] == list(range(389,450))
    data = np.load(HERE / 'plotted_arrays.npz'); old = np.load(OLD / 'plotted_arrays.npz')
    checked = 0
    for key in data:
        if ':mlp' in key or ':transformer' in key:
            np.testing.assert_array_equal(data[key], old[key]); checked += data[key].size
    records = {r['name']: r for r in result['stages']}
    fitted = {n: np.array(g)[1:,1:] for n,g in result['temporal_grams'].items()}
    eigen = {}
    for name, gram in fitted.items():
        values, u = np.linalg.eigh((gram+gram.T)/2)
        values, u = np.maximum(values[::-1],0), u[:,::-1]
        np.testing.assert_allclose(np.linalg.svd(gram,compute_uv=False), values, rtol=2e-8,atol=max(1,values[0])*1e-10)
        spectrum = values/values.sum(); eigen[name] = u
        np.testing.assert_allclose(spectrum,result['curves'][name]['spectrum'],rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(records[name]['top1'],spectrum[0])
        np.testing.assert_allclose(records[name]['top10'],spectrum[:10].sum())
        if name in ['output.log_probabilities','raw_gradient']:
            np.testing.assert_allclose(data[f'spectrum:resnet:{name}:y'],np.cumsum(spectrum))
            np.testing.assert_array_equal(data[f'spectrum:resnet:{name}:x'],np.arange(1,62))
        checked += len(spectrum)
    out = 'output.log_probabilities'
    for name, u in eigen.items():
        overlap = np.linalg.norm(u[:,:10].T@eigen[out][:,:10],ord='fro')**2/10
        np.testing.assert_allclose(overlap,records[name]['temporal_top10_overlap'],rtol=1e-9,atol=1e-11)
    names = [r['name'] for r in result['stages'] if r['name'] != 'raw_gradient']
    for key in ['top1','top10','temporal_top10_overlap']:
        np.testing.assert_array_equal(data[f'spectral-stages:resnet:{key}:x'],[records[n][key] for n in names])
        np.testing.assert_array_equal(data[f'spectral-stages:resnet:{key}:y'],np.arange(6))
        checked += len(names)
    values, u = np.linalg.eigh(fitted[out]); alpha=u[:,-1]*np.sqrt(values[-1])
    if alpha[np.argmax(abs(alpha))]<0: alpha=-alpha
    np.testing.assert_allclose(data['output-coefficient:resnet:y'],alpha/np.sqrt(records[out]['coordinates']))
    np.testing.assert_array_equal(data['output-coefficient:resnet:x'],np.arange(390,451)); checked += len(alpha)
    preservation=read(HERE/'preservation.json')
    for path,digest in preservation['files'].items(): assert sha(ROOT/path)==digest,path
    plan=(ROOT/'manuscript/consolidation/PLAN.md').read_bytes()
    section=plan[plan.index(b'### C. Intervention'):plan.index(b'## VI.')]
    assert hashlib.sha256(section).hexdigest()==preservation['plan_intervention_sha256']
    pdf=HERE/'figures'/NAME
    with fitz.open(pdf) as doc,fitz.open(OLD/'figures'/NAME) as previous:
        assert len(doc)==1
        page=doc[0]; assert '390' in page.get_text() and '450' in page.get_text()
        for font in page.get_fonts(full=True): assert doc.extract_font(font[0])[3]
        for word in page.get_text('words'):
            assert -.1<=word[0]<word[2]<=page.rect.width+.1
            assert -.1<=word[1]<word[3]<=page.rect.height+.1
        width,height=page.rect.width,page.rect.height
        for clip in [fitz.Rect(0,.07*height,.36*width,.90*height),fitz.Rect(.69*width,.07*height,width,.90*height)]:
            assert page.get_pixmap(matrix=fitz.Matrix(2,2),clip=clip).samples==previous[0].get_pixmap(matrix=fitz.Matrix(2,2),clip=clip).samples
    record=dict(status='passed' if visual else 'numerically_verified_awaiting_visual_review',
        verifier_sha256=sha(__file__),pdf_sha256=sha(pdf),checked_values=checked,
        full_window_refit=True,independent_Gram_eigen_and_SVD=True,source_hashes=True,
        mlp_transformer_values_and_panel_pixels_unchanged=True,preserved_files=len(preservation['files']),
        intervention_files_and_PLAN_section_unchanged=True,embedded_fonts=True,text_inside_page=True,
        visual_reviewed=visual,visual_notes='Poppler render inspected: updated61-mode ResNet spectrum, six stage profiles and signed coefficient390--450; no clipped labels or overlaps.' if visual else None)
    (HERE/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--visual-reviewed',action='store_true'); main(parser.parse_args().visual_reviewed)
