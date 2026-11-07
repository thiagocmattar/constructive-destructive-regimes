"""Independently check the seven-row overview against frozen Run215 records."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    manifest = read(HERE / 'optimizer_overview_manifest.json')
    for path, digest in manifest['sources'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in manifest['outputs'].items():
        assert sha(HERE / path) == digest, path
    assert sha(ROOT / manifest['generator']) == manifest['generator_sha256']
    with (HERE / 'optimizer_overview_values.csv').open(newline='') as stream:
        plotted = list(csv.DictReader(stream))
    previous = read(ROOT / 'analyses/055-2026-11-05-adam-directional-amplification/plot_manifest.json')
    expected_keys = {'loss', 'Q', 'rho', 'relative_norm', 'moment_norm', 'moment_rho', 'adaptive_rho'}
    assert len(manifest['curves']) == 21
    assert {c['arm'] for c in manifest['curves']} == {'mlp:0.001', 'resnet:0.003', 'transformer:0.003'}
    checked = preserved = 0
    summaries = {}
    for curve in manifest['curves']:
        folder = ROOT / curve['source']
        boundaries, metrics = [[json.loads(line) for line in (folder / name).read_text().splitlines()]
                               for name in ['boundaries.jsonl', 'metrics.jsonl']]
        assert [b['state'] for b in boundaries] == list(range(1025))
        assert [r['gradient_state'] for r in metrics] == list(range(1024))
        assert [r['update'] for r in metrics] == list(range(1, 1025))
        key = curve['key']
        rows = [r for r in plotted if r['arm'] == curve['arm'] and r['metric'] == key]
        assert len(rows) == curve['points']
        first = 1 if key in {'Q', 'rho', 'moment_rho', 'adaptive_rho'} else 0
        stop = 1025 if key == 'loss' else 1024
        assert [int(r['state']) for r in rows] == list(range(first, stop))
        for row in rows:
            s, actual = int(row['state']), float(row['value'])
            if key == 'loss':
                expected = boundaries[s]['loss']
            elif key == 'relative_norm':
                expected = boundaries[s]['gradient_norm'] / boundaries[0]['gradient_norm']
            elif key == 'Q':
                expected = metrics[s]['projected_gradient_change']
                reconstructed = 1 - metrics[s]['streams']['raw']['cosine'] * boundaries[s]['gradient_norm'] / boundaries[s-1]['gradient_norm']
                assert np.isclose(actual, reconstructed, rtol=2e-12, atol=2e-12)
            else:
                stream, field = {'rho': ('raw', 'cosine'), 'moment_norm': ('moment', 'norm'),
                                 'moment_rho': ('moment', 'cosine'), 'adaptive_rho': ('adaptive', 'cosine')}[key]
                expected = metrics[s]['streams'][stream][field]
                if field == 'cosine':
                    assert metrics[0]['streams'][stream]['cosine'] is None
                    assert -1 <= actual <= 1
            assert actual == expected, (curve['arm'], key, s)
            checked += 1
        for column, field in [('state', 'x_sha256'), ('value', 'y_sha256')]:
            digest = hashlib.sha256(np.asarray([float(r[column]) for r in rows], dtype='<f8').tobytes()).hexdigest()
            assert digest == curve[field]
        if key in {'loss', 'Q', 'rho', 'relative_norm'}:
            old = next(c for c in previous['curves'] if c['figure'] == 'adam_full_trajectories_high.pdf'
                       and c['arm'] == curve['arm'] and c['key'] == key)
            assert all(curve[f] == old[f] for f in ['points', 'x_sha256', 'y_sha256'])
            preserved += len(rows)
        summaries.setdefault(curve['arm'], {})[key] = dict(points=len(rows),
            minimum=min(float(r['value']) for r in rows), maximum=max(float(r['value']) for r in rows))
    assert checked == len(plotted) == 21495
    assert all(set(s) == expected_keys for s in summaries.values())
    pdf = next(p for p in manifest['outputs'] if p.endswith('.pdf'))
    with fitz.open(HERE / pdf) as doc:
        assert len(doc) == 1
        page = doc[0]
        assert abs(page.rect.width / 72 - 7.16) < 1e-5
        assert abs(page.rect.height / 72 - 9.5) < 1e-5
        assert not page.get_images()
        assert len(page.get_drawings()) > 21
        for word in page.get_text('words'):
            assert page.rect.contains(fitz.Rect(word[:4]))
        for font in page.get_fonts(full=True):
            assert font[2] != 'Type3' and doc.extract_font(font[0])[3]
    report = dict(status='passed', verifier_sha256=sha(__file__), manifest_sha256=sha(HERE / 'optimizer_overview_manifest.json'),
                  checked_values=checked, original_row_values_preserved=preserved,
                  added_values=checked-preserved, summaries=summaries,
                  pdf='One page, 7.16 x 9.5 inches, embedded fonts, vector curves, text within bounds',
                  visual_review='See observations/007-optimizer-overview.md')
    (HERE / 'optimizer_overview_verification.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'PASS: {checked:,} values; {preserved:,} original values preserved; {checked-preserved:,} new values checked.')


if __name__ == '__main__':
    main()
