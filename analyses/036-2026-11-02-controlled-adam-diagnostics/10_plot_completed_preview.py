"""Preview only complete, verified Run215 arms in the approved eight-row layout."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/215-2026-11-02-controlled-full-batch-adam'
WINDOW = 16
TASKS = ['mlp', 'resnet', 'transformer']
TITLES = ['CIFAR-10 / MLP', 'CIFAR-10 / ResNet-32', 'WikiText-2 / Transformer']
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rate_label(rate):
    exponent = int(np.floor(np.log10(rate)))
    coefficient = int(round(rate/10**exponent))
    return rf'$\eta={"" if coefficient == 1 else str(coefficient)+r"\times"}10^{{{exponent}}}$'


def main(output_stem='adam_diagnostics_completed_preview', manifest_name='completed_preview_manifest.json'):
    results = read(RUN/'results.json')
    assert results['status'] in ['partial_verified', 'complete_verified']
    config = read(RUN/'config.json')
    updates = config['updates']
    assert updates == 1024
    assert results['completed_arms'] == len(results['conditions'])
    counts = {task: sum(r['task'] == task for r in results['conditions']) for task in TASKS}
    manifest = dict(generator=Path(__file__).relative_to(ROOT).as_posix(),
        generator_sha256=sha(__file__), sources={}, window=WINDOW,
        smoothing='Training loss and cumulative Adam R_t are unsmoothed; the six diagnostic rows use complete 16-observation trailing arithmetic means.',
        clock='Loss states 0..1024; vector measurements associated with applied updates 1..1024.',
        cohort='Complete, locally retrieved, scientifically verified Run215 arms only.',
        completed_counts=counts, omitted_conditions=results['missing'],
        color_policy='Fixed LR-to-color map from the full three-rate grid in config.json.',
        terminal_gradient='Measured terminal gradient is unapplied and excluded from vector rows.',
        rows=['training_loss_log', 'eta_kappa_hat_raw', 'rho_raw', 'rho_momentum',
              'rho_adam', 'raw_gradient_norm_log', 'momentum_norm_log', 'R_t'],
        retention_streams=['actual'], retention_definition='R_t of measured Adam parameter updates.', curves=[],
        limitations=['One fixed initialization per case; no seed aggregation or uncertainty band.',
            'The raw-gradient eta*kappa_hat expression is evaluated along Adam steps; no Hessian or Adam stability threshold is measured.',
            'Smoothing delays and broadens changes; incomplete starting windows are omitted.'])

    def source(path):
        manifest['sources'][path.relative_to(ROOT).as_posix()] = sha(path)

    source(RUN/'results.json')
    source(RUN/'config.json')
    source(RUN/'input_manifest.json')
    for name in ['gd_loss_projected_change.pdf', 'gd_reversal_norm_retention.pdf']:
        source(ROOT/'manuscript/draft/figures/analysis024'/name)

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'path.simplify': False})
    fig, axes = plt.subplots(8, 3, figsize=(7.16, 12.1), sharex=True)
    projected_extrema = []

    def curve(ax, x, values, task, lr, row, stream, smooth=True, **kwargs):
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        assert len(x) == len(y)
        valid = np.isfinite(y)
        if smooth:
            # Only complete, consecutive windows; no padding, extrapolation or filling.
            blocks = np.lib.stride_tricks.sliding_window_view(y, WINDOW)
            full = np.isfinite(blocks).all(axis=1)
            plotted_x = x[WINDOW-1:][full]
            plotted_y = blocks[full].mean(axis=1)
            # Independent convolution check on the finite contiguous source history.
            assert np.flatnonzero(valid).tolist() == list(range(int(np.flatnonzero(valid)[0]), len(y)))
            reference = np.convolve(y[valid], np.ones(WINDOW)/WINDOW, mode='valid')
            np.testing.assert_allclose(plotted_y, reference, rtol=3e-14, atol=3e-14)
        else:
            plotted_x, plotted_y = x[valid], y[valid]
        assert np.isfinite(plotted_y).all()
        artist, = ax.plot(plotted_x, plotted_y, **kwargs)
        np.testing.assert_array_equal(artist.get_xdata(), plotted_x)
        np.testing.assert_array_equal(artist.get_ydata(), plotted_y)
        manifest['curves'].append(dict(task=task, lr=lr, row=row, stream=stream,
            smoothing_window=WINDOW if smooth else None,
            source_x=x.tolist(), source_y=[None if not np.isfinite(v) else float(v) for v in y],
            x=plotted_x.tolist(), y=plotted_y.tolist()))
        return plotted_y

    for col, task in enumerate(TASKS):
        subset = sorted([r for r in results['conditions'] if r['task'] == task], key=lambda r:r['lr'])
        assert 0 <= len(subset) <= 3
        axes[0,col].set_title(TITLES[col], pad=30)
        for condition in subset:
            color = COLORS[config['learning_rates_by_task'][task].index(condition['lr'])]
            folder = ROOT/condition['source']
            identity = read(folder/'manifest.json')
            assert sha(folder/'manifest.json') == condition['manifest_sha256']
            assert identity['status'] == 'complete' and identity['completed_updates'] == updates
            source(folder/'manifest.json')
            for name in ['boundaries.jsonl', 'metrics.jsonl']:
                expected = identity['files'][name]
                assert (folder/name).stat().st_size == expected['bytes'] and sha(folder/name) == expected['sha256']
                source(folder/name)
            boundaries = [json.loads(line) for line in (folder/'boundaries.jsonl').read_text().splitlines()]
            metrics = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            assert [r['state'] for r in boundaries] == list(range(updates+1))
            assert [r['update'] for r in metrics] == list(range(1,updates+1))
            lr = condition['lr']; steps = np.arange(1,updates+1)
            options = dict(color=color, lw=.85)
            curve(axes[0,col], np.arange(updates+1), [r['loss'] for r in boundaries],
                task, lr, 'training_loss_log', 'training', smooth=False, label=rate_label(lr), **options)
            projected = [r['projected_gradient_change'] for r in metrics]
            raw = [r['streams']['raw'] for r in metrics]
            for i in range(1,updates):
                expression = 1-raw[i]['cosine']*raw[i]['norm']/raw[i-1]['norm']
                assert abs(expression-projected[i]) < 2e-12
            projected_extrema.extend(curve(axes[1,col], steps, projected,
                task, lr, 'eta_kappa_hat_raw', 'raw', **options))
            for row, stream, label in [(2,'raw','rho_raw'), (3,'moment','rho_momentum'), (4,'adaptive','rho_adam')]:
                curve(axes[row,col], steps, [r['streams'][stream]['cosine'] for r in metrics],
                    task, lr, label, stream, **options)
            for row,stream,label in [(5,'raw','raw_gradient_norm_log'),(6,'moment','momentum_norm_log')]:
                curve(axes[row,col], steps, [r['streams'][stream]['norm'] for r in metrics],
                    task, lr, label, stream, **options)
            values = [r['streams']['actual']['B_over_A'] for r in metrics]
            assert all(0 <= value <= 1+1e-10 for value in values)
            curve(axes[7,col], steps, values, task, lr, 'R_t', 'actual', smooth=False, **options)
        if subset:
            axes[0,col].legend(loc='lower center', bbox_to_anchor=(.5,1.01), ncol=3,
                fontsize=7, frameon=False, columnspacing=.7, handlelength=1.1, handletextpad=.25)
        else:
            axes[0,col].text(.5,1.06,'No completed runs yet',transform=axes[0,col].transAxes,
                ha='center',va='bottom',fontsize=8,color='.4')
        for row in [0,5,6]:
            axes[row,col].set_yscale('log')
            axes[row,col].yaxis.set_minor_formatter(NullFormatter())
        norm_limits = (min(axes[row,col].get_ylim()[0] for row in [5,6]),
                       max(axes[row,col].get_ylim()[1] for row in [5,6]))
        for row in [5,6]:
            axes[row,col].set_ylim(norm_limits)
        axes[0,col].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axes[0,col].yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
        for row in [2,3,4]:
            axes[row,col].set(ylim=(-1.08,1.08), yticks=[-1,0,1])
            axes[row,col].axhline(0, color='.5', lw=.6, zorder=0)
        axes[1,col].axhline(2, color='.35', lw=.7, ls='--', zorder=0)
        axes[7,col].set(ylim=(0,1.04), yticks=[0,.5,1], xlabel='Training step')
        for row in range(8):
            ax = axes[row,col]
            ax.set(xlim=(0,updates), xticks=[0,256,512,768,1024])
            ax.text(0,1.035,f'({chr(97+row*3+col)})',transform=ax.transAxes,va='bottom',fontsize=8)
            ax.tick_params(length=3,width=.6)
            ax.grid(axis='y',alpha=.15,lw=.5)
    low, high = min(-.08,min(projected_extrema)), max(2.18,max(projected_extrema)*1.04)
    for ax in axes[1]:
        ax.set(ylim=(low,high), yticks=[0,1,2])
    labels = ['Training loss', r'$\eta\widehat\kappa_t$'+'\n(raw gradients)',
        r'$\rho_t$'+'\n(raw gradients)', r'$\rho_t$'+'\n(momentum)',
        r'$\rho_t$'+'\n(Adam)', r'$\|g_t\|$', r'$\|m_t\|$', r'$R_t$'+'\n(Adam)']
    for ax,label in zip(axes[:,0],labels):
        ax.set_ylabel(label)
    fig.subplots_adjust(left=.105,right=.975,top=.925,bottom=.10,hspace=.34,wspace=.29)
    fig.text(.545,.018,'Training loss and Adam '+r'$R_t$'+' unsmoothed; other rows: 16-step trailing means.',
        ha='center',fontsize=8)
    fig.text(.545,.045,
        f'Completed runs only: MLP {counts["mlp"]}/3, ResNet-32 {counts["resnet"]}/3, Transformer {counts["transformer"]}/3.',
        ha='center',fontsize=8)
    for col, task in enumerate(TASKS):
        if counts[task] == 0:
            for ax in axes[:,col]:
                ax.set_yticks([]); ax.set_yticks([],minor=True)
                ax.set_facecolor('#f7f7f7'); ax.grid(False)
                for line in ax.lines: line.set_visible(False)
    output = HERE/'figures'/(output_stem+'.pdf')
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert len(manifest['curves']) == 8*results['completed_arms']
    manifest['output'] = output.relative_to(ROOT).as_posix()
    manifest['output_sha256'] = sha(output)
    manifest['status'] = 'numerical_checks_passed'
    (HERE/manifest_name).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(output=manifest['output'], curves=len(manifest['curves']), window=WINDOW)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-stem', default='adam_diagnostics_completed_preview')
    parser.add_argument('--manifest-name', default='completed_preview_manifest.json')
    args = parser.parse_args()
    main(args.output_stem, args.manifest_name)
