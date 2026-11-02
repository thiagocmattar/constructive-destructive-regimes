"""First-64-lag version of the verified largest-rate GD figure."""
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEM = 'largest_lr_lag64'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    prior_path = HERE/'largest_lr_lag256_review.json'
    prior = json.loads(prior_path.read_text())
    assert prior['status'] == prior['visual_review']['status'] == 'passed'
    assert sha(HERE/'results.json') == prior['source_results_sha256']
    assert sha(HERE/'lag_profiles.csv') == prior['source_csv_sha256']
    profiles = {(p['task'], p['anchor']): p
                for p in json.loads((HERE/'results.json').read_text())['profiles']
                if p['lr'] == prior['rates'][p['task']]}
    selected = {(p['task'], p['anchor']): p for p in prior['profiles']}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'path.simplify': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.8), layout='constrained')
    lines = []
    cases = [('mlp', 'CIFAR-10 / MLP'), ('resnet', 'CIFAR-10 / ResNet-32'),
             ('transformer', 'WikiText-2 / Transformer')]
    for i, anchor in enumerate([500, 2000, 3000]):
        for j, (task, title) in enumerate(cases):
            p = profiles[(task, anchor)]
            assert p['primary_eligible'] and p['parent_run_id'] == selected[(task, anchor)]['parent_run_id']
            rows = [r for r in p['rows'] if 1 <= r['lag'] <= 64]
            x = np.array([r['lag'] for r in rows])
            y = np.array([r['cosine'] for r in rows])
            assert np.array_equal(x, np.arange(1, 65)) and np.isfinite(y).all()
            ax = axes[i, j]
            artist, = ax.plot(x, y, color='#246b9a', linewidth=.7, drawstyle='default')
            assert np.array_equal(artist.get_xdata(), x) and np.array_equal(artist.get_ydata(), y)
            assert len(ax.lines) == 1
            ax.axhline(0, color='.55', linewidth=.65, zorder=0)
            ax.set(xlim=(1, 64), ylim=(-1.05, 1.05), xticks=[1, 16, 32, 48, 64],
                   yticks=[-1, -.5, 0, .5, 1])
            if i == 0:
                ax.set_title(f'{title}\nGD learning rate {p["lr"]:g}', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'Anchor {anchor:,}\ncos(g₀, gₖ)')
            if i == 2:
                ax.set_xlabel('Lag k (GD updates after anchor)')
            lines.append(dict(task=task, parent_run_id=p['parent_run_id'], anchor=anchor,
                              lr=p['lr'], points=len(rows), exact_source_artist_match=True))
    fig.suptitle('Anchored raw-gradient cosine at the largest GD learning rate', fontsize=14)
    fig.supxlabel('All lags 1–64 connected in order; each row starts from its retained anchor.', fontsize=9)
    pdf = HERE/'figures'/f'{STEM}.pdf'
    fig.savefig(pdf, bbox_inches='tight', metadata={'Creator': Path(__file__).name})
    plt.close(fig)
    png = HERE/'scratch'/f'{STEM}.png'
    png.parent.mkdir(exist_ok=True)
    with fitz.open(pdf) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for block in doc[0].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    assert doc[0].rect.contains(fitz.Rect(span['bbox']))
        doc[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(png)
    record = dict(status='passed', generator=Path(__file__).name, generator_sha256=sha(Path(__file__)),
                  source_results_sha256=prior['source_results_sha256'], source_csv_sha256=prior['source_csv_sha256'],
                  source_lag256_review_sha256=sha(prior_path), profiles=lines, plotted_points=576,
                  reduction='Exact first-64-lag subset of the verified 256-lag view; all lags connected in order, no parity split, smoothing or downsampling.',
                  figure=pdf.relative_to(HERE).as_posix(), figure_sha256=sha(pdf),
                  render=png.relative_to(HERE).as_posix(), render_sha256=sha(png),
                  vector_pdf=True, text_within_page=True, visual_review='pending actual-PDF inspection')
    assert len(lines) == 9 and sum(line['points'] for line in lines) == record['plotted_points']
    (HERE/f'{STEM}_review.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(profiles=len(lines), points=record['plotted_points'], figure=record['figure'])))


if __name__ == '__main__':
    main()
