"""Replace only Figure 6's ResNet window using existing untreated captures."""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'figure_revision_20261114'
NAME = '00-three-model-loss-representation-motion.pdf'
SOURCE = ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing/resnet_390_450_motion.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main():
    spec = importlib.util.spec_from_file_location('original_figure_style', OLD / '01_figures.py')
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    v.OUT = HERE / 'figures'
    v.OUT.mkdir(exist_ok=True)
    preservation = HERE / 'preservation.json'
    if not preservation.exists():
        evidence = ROOT / 'manuscript/consolidation/evidence'
        keep = list((evidence / '05-adam/c-intervention').rglob('*'))
        keep += [p for p in evidence.rglob('*.pdf') if p.name != NAME]
        keep = sorted(set(p for p in keep if p.is_file()))
        plan = (ROOT / 'manuscript/consolidation/PLAN.md').read_bytes()
        section = plan[plan.index(b'### C. Intervention'):plan.index(b'## VI.')]
        write(preservation, dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in keep},
                                 plan_intervention_sha256=hashlib.sha256(section).hexdigest()))

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 9,
                         'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                         'pdf.fonttype': 42, 'path.simplify': False})
    data = v.trace_data()
    r = v.read(SOURCE)
    assert r['status'] == 'verified'
    x = np.array(r['completed_steps'])
    rho = np.array([s['rho'] for s in r['stages']])
    rms = np.array([s['change_rms'] for s in r['stages']])
    record = dict(case='resnet', source_outgoing_window=r['source_outgoing_window'],
                  displayed_completed_steps=[int(x[0]), int(x[-1])], **r['event'],
                  stages=[{'source': s['name'], 'label': s['label']} for s in r['stages']],
                  rms_window_maxima=rms.max(axis=1).tolist(), loss_threshold=1.1,
                  heatmap_values=int(rho.size + rms.size), rho_sha256=v.arrsha(rho), rms_sha256=v.arrsha(rms))
    v.AUDIT['tracing'][1] = record
    data['resnet'] = dict(x=x, loss=np.array(r['loss']), probe=np.array(r['probe_loss']),
                          names=[s['name'] for s in r['stages']], labels=[s['label'] for s in r['stages']],
                          rho=rho, relative=rms / rms.max(axis=1)[:, None], record=record)

    fig, axes = plt.subplots(3, 3, figsize=(7.16, 4.65), sharex='col',
                             gridspec_kw={'height_ratios': [1.1, 1.45, 1.45]})
    for col, case in enumerate(v.CASES):
        d = data[case]
        x, record = d['x'], d['record']
        extent = [x[0] - .5, x[-1] + .5, len(d['names']) - .5, -.5]
        v.curve(axes[0, col], x, d['loss'], f'trace:{case}:full-loss', color='.12', lw=1)
        if case != 'mlp':
            v.curve(axes[0, col], x, d['probe'], f'trace:{case}:probe-loss', color=v.COLORS[1], lw=.9, ls='--')
        axes[0, col].set_title(f'({chr(97 + col)}) {v.TITLES[col]}', pad=8)
        axes[0, col].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        v.style(axes[0, col])
        crossing = record['loss_threshold_step']
        axes[0, col].plot(crossing, d['loss'][np.where(x == crossing)[0][0]], 'o', color='.12', ms=2.6, zorder=6)
        for row, values, cmap, vmin in [(1, d['rho'], 'RdBu_r', -1), (2, d['relative'], 'viridis', 0)]:
            im = axes[row, col].imshow(values, aspect='auto', interpolation='nearest', extent=extent,
                                       cmap=cmap, vmin=vmin, vmax=1)
            np.testing.assert_array_equal(np.asarray(im.get_array()), values)
            v.PLOTTED[f'heatmap:{case}:{row}'] = values
            axes[row, col].set_yticks(range(len(d['labels'])), d['labels'], fontsize=7.5)
            axes[row, col].tick_params(length=0, pad=2)
            v.letter(axes[row, col], chr(97 + 3 * row + col))
            if row == 1:
                rho_image = im
            else:
                rms_image = im
        ticks = {'mlp': [451, 465, 480, 501], 'resnet': [390, 405, 420, 450],
                 'transformer': [553, 565, 580, 601]}[case]
        for ax in axes[:, col]:
            ax.set(xlim=(x[0] - .5, x[-1] + .5), xticks=ticks)
            v.markers(ax, record)
        axes[-1, col].set_xlabel(v.XLABEL)
    axes[0, 0].set_ylabel('Training loss')
    axes[1, 0].set_ylabel('Change cosine', labelpad=9)
    axes[2, 0].set_ylabel('Displacement RMS\n/ stage maximum', labelpad=7)
    handles = [Line2D([], [], color='.12', lw=1, label='Full objective'),
               Line2D([], [], color=v.COLORS[1], ls='--', label='Fixed probe'),
               Line2D([], [], color=v.COLORS[0], ls='--', label=r'First observed output $\rho<0$'),
               Line2D([], [], color='.15', ls='--', label='Loss +10%')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .005), ncol=2,
               frameon=False, fontsize=7.5, columnspacing=1.4, handlelength=2)
    fig.subplots_adjust(left=.14, right=.91, top=.91, bottom=.20, hspace=.31, wspace=.55)
    box = axes[1, 2].get_position()
    fig.colorbar(rho_image, cax=fig.add_axes([.932, box.y0, .012, box.height]), ticks=[-1, 0, 1])
    box = axes[2, 2].get_position()
    fig.colorbar(rms_image, cax=fig.add_axes([.932, box.y0, .012, box.height]), ticks=[0, .5, 1])
    v.save(fig, NAME)
    np.savez_compressed(HERE / 'plotted_arrays.npz', **v.PLOTTED)
    manifest = dict(status='numerical_checks_passed', generator_sha256=sha(__file__),
                    style_generator_sha256=sha(OLD / '01_figures.py'), sources=v.AUDIT['sources'],
                    curves=v.AUDIT['curves'], figures=v.AUDIT['figures'], tracing=v.AUDIT['tracing'],
                    plotted_arrays_sha256=sha(HERE / 'plotted_arrays.npz'),
                    scope='Only the ResNet column changes. Original MLP/Transformer arrays and all other PDFs remain unchanged.',
                    svd_scope='The separate temporal-structure PDF retains the original ResNet 281--311 completed-step window; no new SVD is inferred.')
    write(HERE / 'figure_manifest.json', manifest)
    print(json.dumps({'figure': NAME, 'resnet_event': r['event'], 'source': str(SOURCE.relative_to(ROOT))}, indent=2))


if __name__ == '__main__':
    main()
