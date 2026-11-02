"""Plot the largest GD rate per architecture, joining every lag in order."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEM = 'largest_lr_lag256'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    profiles = json.loads((HERE/'results.json').read_text())['profiles']
    tasks = ['mlp', 'resnet', 'transformer']
    titles = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
    rates = {task: max(p['lr'] for p in profiles if p['task'] == task) for task in tasks}
    assert rates == dict(mlp=.04, resnet=.1, transformer=.2)
    with (HERE/'lag_profiles.csv').open(newline='', encoding='utf-8') as stream:
        independent = {(r['parent_run_id'], int(r['anchor']), int(r['lag'])): r
                       for r in csv.DictReader(stream)}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'path.simplify': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.8), layout='constrained')
    lines = []
    for i, anchor in enumerate([500, 2000, 3000]):
        for j, (task, title) in enumerate(zip(tasks, titles)):
            matches = [p for p in profiles if p['task'] == task
                       and p['lr'] == rates[task] and p['anchor'] == anchor]
            assert len(matches) == 1
            profile = matches[0]
            assert profile['primary_eligible'] and profile['last_lag'] >= 256
            rows = [r for r in profile['rows'] if 1 <= r['lag'] <= 256]
            x = np.array([r['lag'] for r in rows])
            y = np.array([r['cosine'] for r in rows])
            assert np.array_equal(x, np.arange(1, 257))
            assert np.isfinite(y).all() and np.max(np.abs(y)) <= 1 + 1e-12
            for row in rows:
                original = independent[(profile['parent_run_id'], anchor, row['lag'])]
                assert float(original['cosine']) == row['cosine']
                assert int(original['state']) == anchor + row['lag']
            ax = axes[i, j]
            artist, = ax.plot(x, y, color='#246b9a', linewidth=.7, drawstyle='default')
            assert np.array_equal(artist.get_xdata(), x)
            assert np.array_equal(artist.get_ydata(), y)
            assert len(ax.lines) == 1  # One ordered curve, before the zero reference.
            ax.axhline(0, color='.55', linewidth=.65, zorder=0)
            ax.set(xlim=(1, 256), ylim=(-1.05, 1.05), xticks=[1, 64, 128, 192, 256],
                   yticks=[-1, -.5, 0, .5, 1])
            if i == 0:
                ax.set_title(f'{title}\nGD learning rate {rates[task]:g}', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'Anchor {anchor:,}\ncos(g₀, gₖ)')
            if i == 2:
                ax.set_xlabel('Lag k (GD updates after anchor)')
            lines.append(dict(task=task, parent_run_id=profile['parent_run_id'],
                              anchor=anchor, lr=rates[task], points=len(rows),
                              plotted_values_sha256=hashlib.sha256(
                                  json.dumps(list(zip(x.tolist(), y.tolist())),
                                             separators=(',', ':')).encode()).hexdigest(),
                              exact_json_csv_artist_match=True))
    fig.suptitle('Anchored raw-gradient cosine at the largest GD learning rate', fontsize=14)
    fig.supxlabel('All lags 1–256 connected in order; each row starts from its retained anchor.', fontsize=9)
    pdf = HERE/'figures'/f'{STEM}.pdf'
    fig.savefig(pdf, bbox_inches='tight', metadata={'Creator': Path(__file__).name})
    plt.close(fig)
    scratch = HERE/'scratch'
    scratch.mkdir(exist_ok=True)
    png = scratch/f'{STEM}.png'
    with fitz.open(pdf) as doc:
        assert len(doc) == 1 and not doc[0].get_images()
        for block in doc[0].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    assert doc[0].rect.contains(fitz.Rect(span['bbox']))
        doc[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(png)
    record = dict(status='passed', generator=Path(__file__).name,
                  generator_sha256=sha(Path(__file__)), source_results_sha256=sha(HERE/'results.json'),
                  source_csv_sha256=sha(HERE/'lag_profiles.csv'), rates=rates, profiles=lines,
                  plotted_points=sum(line['points'] for line in lines),
                  reduction='Select largest GD LR per architecture; retain every lag 1..256 in order. No parity split, smoothing, averaging, or downsampling.',
                  figure=pdf.relative_to(HERE).as_posix(), figure_sha256=sha(pdf),
                  render=png.relative_to(HERE).as_posix(), render_sha256=sha(png),
                  vector_pdf=True, text_within_page=True,
                  visual_review='pending inspection of actual PDF render')
    assert record['plotted_points'] == 2304 and len(lines) == 9
    (HERE/f'{STEM}_review.json').write_text(json.dumps(record, indent=2)+'\n',
                                         encoding='utf-8', newline='\n')
    print(json.dumps(dict(profiles=len(lines), points=record['plotted_points'],
                          rates=rates, figure=record['figure'])))


if __name__ == '__main__':
    main()
