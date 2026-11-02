"""Independently verify the two complete nine-arm Adam figure reductions."""
import hashlib
import json
import math
from pathlib import Path

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    basefile=HERE/'full_diagnostics_manifest.json'
    rawfile=HERE/'full_unsmoothed_manifest.json'
    base=json.loads(basefile.read_text());raw=json.loads(rawfile.read_text())
    assert base['window']==50
    assert base['completed_counts']==raw['completed_counts']==dict(mlp=3,resnet=3,transformer=3)
    assert len(base['curves'])==len(raw['curves'])==72
    assert sha(basefile)==raw['source_preview_sha256']
    for path,digest in base['sources'].items():assert sha(ROOT/path)==digest
    maximum=0.;base_points=0;raw_points=0;conditions=set()
    for smoothed,unsmoothed in zip(base['curves'],raw['curves']):
        assert all(smoothed[k]==unsmoothed[k] for k in ['task','lr','row','stream'])
        conditions.add((smoothed['task'],smoothed['lr']))
        pairs=[(x,y) for x,y in zip(smoothed['source_x'],smoothed['source_y'])
            if y is not None and math.isfinite(y)]
        assert unsmoothed['x']==[x for x,y in pairs]
        assert unsmoothed['y']==[y for x,y in pairs] and unsmoothed['smoothing_window'] is None
        raw_points+=len(pairs)
        if smoothed['row'] in ['training_loss_log','R_t']:
            assert smoothed['smoothing_window'] is None
            assert smoothed['x']==unsmoothed['x'] and smoothed['y']==unsmoothed['y']
            if smoothed['row']=='training_loss_log':assert smoothed['x']==list(range(1025))
            else:assert smoothed['stream']=='actual' and smoothed['x']==list(range(1,1025))
        else:
            assert smoothed['smoothing_window']==50
            expected_x=[];expected_y=[];ys=smoothed['source_y'];xs=smoothed['source_x']
            for k in range(49,len(ys)):
                values=ys[k-49:k+1]
                if all(v is not None and math.isfinite(v) for v in values):
                    expected_x.append(xs[k]);expected_y.append(math.fsum(values)/50)
            assert smoothed['x']==expected_x
            np.testing.assert_allclose(smoothed['y'],expected_y,rtol=3e-14,atol=3e-14)
            maximum=max(maximum,max(abs(x-y) for x,y in zip(smoothed['y'],expected_y)))
        if unsmoothed['row']=='eta_kappa_hat_raw':
            assert min(unsmoothed['y'])>=raw['projected_change_ylim'][0]
            assert max(unsmoothed['y'])<=raw['projected_change_ylim'][1]
        base_points+=len(smoothed['y'])
    assert len(conditions)==9
    figures=[]
    for manifest in [base,raw]:
        assert sha(ROOT/manifest['generator'])==manifest['generator_sha256']
        pdf=ROOT/manifest['output'];assert sha(pdf)==manifest['output_sha256']
        doc=fitz.open(pdf);assert len(doc)==1;page=doc[0];text=page.get_text()
        assert all(s in text for s in ['MLP 3/3','ResNet-32 3/3','Transformer 3/3'])
        if manifest is base:assert '50-step trailing means.' in text
        for word in page.get_text('words'):
            assert word[0]>=0 and word[1]>=0 and word[2]<=page.rect.width+.01 and word[3]<=page.rect.height+.01,word
        labels=page.search_for('Training step');footer=page.search_for('Completed runs only:')
        assert len(labels)==3 and len(footer)==1 and all(p.y1+3<footer[0].y0 for p in labels)
        figures.append(dict(path=manifest['output'],sha256=manifest['output_sha256'],pages=1,text_bounds_inside_page=True))
    record=dict(status='numerical_and_pdf_checks_passed',conditions=9,curves_per_figure=72,
        diagnostic_smoothing_window=50,
        smoothed_diagnostic_figure_points=base_points,fully_unsmoothed_figure_points=raw_points,
        independent_mean_max_absolute_error=maximum,raw_arrays_and_indices_exact=True,
        loss_and_Adam_R_t_unsmoothed_and_identical_between_versions=True,
        all_nine_conditions_identical_between_versions=True,all_source_hashes_verified=True,
        full_manifest_sha256=sha(basefile),raw_manifest_sha256=sha(rawfile),figures=figures,
        visual_review='pending_final_renders')
    (HERE/'full_diagnostics_verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':main()
