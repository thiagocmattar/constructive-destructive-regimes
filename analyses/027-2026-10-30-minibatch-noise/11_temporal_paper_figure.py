"""Resolve the state-1000 batch comparison using retained, hash-checked histories."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production'
BATCHES = [32, 2048, 5000]
WINDOW = 20
BLUE, ORANGE = '#0072B2', '#D55E00'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    verified = read(HERE / 'verification.json')
    assert verified['status'] == 'passed'
    assert digest(RAW / 'inventory.json') == verified['production_inventory_sha256']
    inventory = {r['path']: r['sha256'] for r in read(RAW / 'inventory.json')['files']}
    manifest = dict(script_sha256=digest(Path(__file__)),
                    production_inventory_sha256=digest(RAW / 'inventory.json'),
                    checkpoint=1000, batches=BATCHES, trailing_pairs=WINDOW,
                    reduction='Average repetitions within initialization, then trailing 20-pair means; equal-weight mean and range of three initializations.',
                    selection='Smallest batch, largest stochastic batch, full batch; all five batches remain in the outcome figure.',
                    raw_sources={}, curves=[], summaries=[], references=[])

    def rows(rel):
        path = RAW / rel
        assert digest(path) == inventory[rel], rel
        manifest['raw_sources'][rel] = inventory[rel]
        records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        assert [r['update'] for r in records] == list(range(1, len(records) + 1))
        return records

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True)
    for col, b in enumerate(BATCHES):
        ax = axes[col]
        streams = {key: [] for key in ['cosine', 'rho_full']}
        events = []
        for seed in range(3):
            repetitions = []
            for rep in range(1 if b == 5000 else 8):
                records = rows(f'moving-seed{seed}-state1000-b{b}-rep{rep:02d}/metrics.jsonl')
                assert len(records) == 200
                repetitions.append(records)
                events.append(dict(seed=seed, replicate=rep,
                    first_negative_sampled=next((r['update'] for r in records if r['cosine'] is not None and r['cosine'] < 0), None),
                    first_negative_full=next((r['update'] for r in records if r['rho_full'] is not None and r['rho_full'] < 0), None)))
            for key in streams:
                a = np.array([[r[key] for r in records[1:]] for records in repetitions])
                assert a.shape == (len(repetitions), 199) and np.isfinite(a).all()
                streams[key].append(a.mean(0))
        summary = dict(batch=b, first_negative=events)
        for key, color in [('cosine', BLUE), ('rho_full', ORANGE)]:
            raw = np.array(streams[key])
            smoothed = np.array([np.convolve(a, np.ones(WINDOW) / WINDOW, mode='valid') for a in raw])
            updates = np.arange(WINDOW + 1, 201)
            assert smoothed.shape == (3, 180)
            center, low, high = smoothed.mean(0), smoothed.min(0), smoothed.max(0)
            ax.fill_between(updates, low, high, color=color, alpha=.12, lw=0)
            line, = ax.plot(updates, center, color=color, lw=1.15)
            np.testing.assert_array_equal(line.get_ydata(), center)
            assert np.max(high) <= 1.00000001 and np.min(low) >= -1.00000001
            manifest['curves'].append(dict(batch=b, field=key, updates=updates.tolist(),
                initialization_means=smoothed.tolist(), mean=center.tolist(),
                minimum=low.tolist(), maximum=high.tolist()))
            summary[key] = dict(initialization_means=raw.mean(1).tolist(), mean=float(raw.mean()),
                                mean_updates_2_to_20=float(raw[:, :19].mean()))
        if b == 5000:
            np.testing.assert_allclose(streams['cosine'], streams['rho_full'], atol=1e-12, rtol=0)
        manifest['summaries'].append(summary)
        ax.axhline(0, color='.5', lw=.6, zorder=0)
        ax.set(xlim=(0, 205), ylim=(-1.08, 1.08), xlabel='Continuation update',
               title=(rf'$b={b:,}$' + (' (full GD)' if b == 5000 else '')))
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.tick_params(length=3, width=.6)
        ax.grid(axis='y', alpha=.18, lw=.5)
        ax.text(0, 1.035, f'({chr(97 + col)})', transform=ax.transAxes, va='bottom', ha='left')
    axes[0].set_ylabel('Mean cosine\nsimilarity ' + r'$\rho_t$')
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.15, label='Mini-batch gradients'),
                        Line2D([], [], color=ORANGE, lw=1.15, label='Full gradients at the same SGD states')],
               loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.095, right=.985, top=.82, bottom=.29, wspace=.2)
    name = 'sgd_alignment_from_state1000.pdf'
    source = HERE / 'figures/paper' / name
    fig.savefig(source, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    target = ROOT / 'manuscript/draft/figures/analysis027' / name
    shutil.copyfile(source, target)
    manifest['pdf'] = dict(source=source.relative_to(ROOT).as_posix(),
                           target=target.relative_to(ROOT).as_posix(), sha256=digest(source))
    for seed in range(3):
        records = rows(f'reference-seed{seed}/metrics.jsonl')
        manifest['references'].append(dict(seed=seed,
            first_negative=next(r['update'] for r in records if r['cosine'] is not None and r['cosine'] < 0)))
    (HERE / 'temporal_figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(pdf=name, curves=len(manifest['curves']), source_files=len(manifest['raw_sources']),
                          plotted_means=1080, initialization_values=3240)))


if __name__ == '__main__':
    main()
