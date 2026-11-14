"""Independent raw-boundary, figure-preservation and PDF checks."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'figure_revision_20261114'
NAME = '00-three-model-loss-representation-motion.pdf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(visual_reviewed):
    m = read(HERE / 'figure_manifest.json')
    assert m['generator_sha256'] == sha(HERE / '01_figure.py')
    assert m['style_generator_sha256'] == sha(OLD / '01_figures.py')
    for path, digest in m['sources'].items():
        assert sha(ROOT / path) == digest, path
    r = read(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_motion.json')
    assert r['generator_sha256'] == sha(ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/11_resnet_major_peak.py')
    for path, digest in r['sources'].items():
        assert sha(ROOT / path) == digest, path
    path = ROOT / 'runs/273-2026-11-07-resnet-distributed-displacement/artifacts/resnet-a/resnet-distributed/control/boundaries.jsonl'
    boundaries = {q['state']: q for q in map(json.loads, path.read_text().splitlines())}
    plotted = np.load(HERE / 'plotted_arrays.npz')
    old = np.load(OLD / 'plotted_arrays.npz')
    checked = 0
    for case in ['mlp', 'resnet', 'transformer']:
        record = next(x for x in m['tracing'] if x['case'] == case)
        x = plotted[f'trace:{case}:full-loss:x']
        if case == 'resnet':
            np.testing.assert_array_equal(x, np.arange(390, 451))
            loss = [boundaries[int(s)]['loss'] for s in x]
            probe = [boundaries[int(s)]['probe_loss'] for s in x]
            rho = np.array([[boundaries[int(s)]['representations'][stage['source']]['rho'] for s in x]
                            for stage in record['stages']])
            rms = np.array([[boundaries[int(s)]['representations'][stage['source']]['change_rms'] for s in x]
                            for stage in record['stages']])
            np.testing.assert_array_equal(plotted['trace:resnet:probe-loss:y'], probe)
            assert tuple(record[k] for k in ['baseline_step', 'first_rise_step', 'loss_threshold_step', 'peak_step', 'recovery_step']) == (404, 405, 406, 415, 439)
        else:
            source = ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing'
            source = source / ('mlp_450_500_results.json' if case == 'mlp' else 'transformer_results.json')
            a = read(source)
            for suffix in ['x', 'y']:
                np.testing.assert_array_equal(plotted[f'trace:{case}:full-loss:{suffix}'], old[f'trace:{case}:full-loss:{suffix}'])
            if case == 'transformer':
                np.testing.assert_array_equal(plotted['trace:transformer:probe-loss:y'], old['trace:transformer:probe-loss:y'])
            loss = old[f'trace:{case}:full-loss:y']
            rho = np.array([a['curves'][stage['source']]['rho'] for stage in record['stages']])
            rms = np.array([a['curves'][stage['source']]['change_rms'] for stage in record['stages']])
        np.testing.assert_array_equal(plotted[f'trace:{case}:full-loss:y'], loss)
        np.testing.assert_array_equal(plotted[f'heatmap:{case}:1'], rho)
        np.testing.assert_array_equal(plotted[f'heatmap:{case}:2'], rms / rms.max(axis=1)[:, None])
        negative = int(np.flatnonzero(rho[-1] < 0)[0])
        assert x[negative] == record['first_observed_output_reversal_step']
        assert (negative == 0) == record['reversal_left_censored']
        base = list(x).index(record['baseline_step'])
        peak = list(x).index(record['peak_step'])
        first = next(i for i in range(base + 1, peak + 1) if loss[i] >= 1.1 * loss[base])
        assert x[first] == record['loss_threshold_step']
        checked += rho.size + rms.size + len(loss) * (1 if case == 'mlp' else 2)

    preservation = read(HERE / 'preservation.json')
    for path, digest in preservation['files'].items():
        assert sha(ROOT / path) == digest, path
    plan = (ROOT / 'manuscript/consolidation/PLAN.md').read_bytes()
    section = plan[plan.index(b'### C. Intervention'):plan.index(b'## VI.')]
    assert hashlib.sha256(section).hexdigest() == preservation['plan_intervention_sha256']
    pdf = HERE / 'figures' / NAME
    assert sha(pdf) == m['figures'][0]['sha256']
    with fitz.open(pdf) as doc, fitz.open(OLD / 'figures' / NAME) as previous:
        assert len(doc) == 1
        page = doc[0]
        assert '390' in page.get_text() and '450' in page.get_text() and page.get_text().count('Training step') == 3
        for font in page.get_fonts(full=True):
            assert doc.extract_font(font[0])[3]
        for word in page.get_text('words'):
            assert -.1 <= word[0] < word[2] <= page.rect.width + .1
            assert -.1 <= word[1] < word[3] <= page.rect.height + .1
        width, height = page.rect.width, page.rect.height
        for clip in [fitz.Rect(0, 0, .36 * width, .81 * height), fitz.Rect(.67 * width, 0, width, .81 * height)]:
            assert page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip).samples == previous[0].get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip).samples
    captions = (HERE / 'captions.tex').read_text(encoding='utf-8')
    assert '390--450' in captions and '281--311' in captions and '311.53' in captions
    result = dict(status='passed' if visual_reviewed else 'numerically_verified_awaiting_visual_review',
                  verifier_sha256=sha(__file__), pdf_sha256=sha(pdf), checked_values=checked,
                  raw_boundary_indexing=True, raw_source_hashes=True, loss_only_event_selection=True,
                  mlp_transformer_arrays_and_panel_pixels_unchanged=True,
                  preserved_files=len(preservation['files']), intervention_section_unchanged=True,
                  embedded_fonts=True, text_inside_page=True, caption_windows_explicit=True,
                  visual_reviewed=visual_reviewed,
                  visual_notes='Poppler rendering reviewed: large ResNet peak, pre-rise and recovery, six aligned stages; no overlaps or clipping.' if visual_reviewed else None)
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed', action='store_true')
    main(parser.parse_args().visual_reviewed)
