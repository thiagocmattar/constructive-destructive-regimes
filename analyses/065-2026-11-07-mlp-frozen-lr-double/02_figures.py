"""Side-by-side cumulative interventions, each against unchanged Adam."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
COLORS = {'control': '#2166ac', 'freeze': '#d95f02', 'double': '#7b3294', 'high': '#c51b2f', 'mid': '#00866a',
          'halve_005':'#b8860b','halve_0025':'#a64d79','halve_00125':'#257a30','halve_000625':'#555555'}
LABELS = {'control': 'Baseline: unchanged Adam', 'freeze': 'I1: freeze linear 1',
          'double': 'I2: freeze + LR 0.002 at 500', 'high': 'I3: freeze + LR 0.02 at 500',
          'mid': 'I4: freeze + LR 0.01 at 500', 'halve_005':'I5: freeze + LR 0.005 at 500',
          'halve_0025':'I6: freeze + LR 0.0025 at 500','halve_00125':'I7: freeze + LR 0.00125 at 500',
          'halve_000625':'I8: freeze + LR 0.000625 at 500'}
ARMS = ['freeze', 'double', 'high', 'mid', 'halve_005', 'halve_0025', 'halve_00125', 'halve_000625']
TITLES = ['I1: freeze linear 1; LR 0.001', 'I2: freeze + LR 0.002 from 500',
          'I3: freeze + LR 0.02 from 500', 'I4: freeze + LR 0.01 from 500',
          'I5: freeze + LR 0.005 from 500', 'I6: freeze + LR 0.0025 from 500',
          'I7: freeze + LR 0.00125 from 500','I8: freeze + LR 0.000625 from 500']
TITLE_BY_ARM = dict(zip(ARMS,TITLES))
NAMES = ['preactivation1', 'hidden1', 'preactivation2', 'hidden2', 'output']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2, lw=.6)


def legend(fig, handles, y, group):
    fig.legend(handles, [LABELS[k] for k in ['control', *group]],
               loc='upper center', bbox_to_anchor=(.53, y), ncol=len(group)+1, frameon=False, fontsize=10)


def main():
    source = HERE/'comparison.json'
    r = json.loads(source.read_text())
    assert r['status'] == 'verified' and set(r['curves']) == {'control', *ARMS}
    for name, digest in r['sources'].items():
        assert sha(ROOT/name) == digest, name
    assert r['generator_sha256'] == sha(HERE/'01_reduce.py')
    (HERE/'figures').mkdir(exist_ok=True)
    target = HERE/'figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf'
    t = np.array(r['states'])
    assert np.array_equal(t, np.arange(448, 801))
    # Shared physical scales include every arm, also across page groups.
    metric_values={0:[],1:[],3:[],4:[],6:[]}
    for c in r['curves'].values():
        for i,key in [(0,'loss'),(1,'eta_kappa_hat'),(3,'raw_gradient_norm'),(4,'active_gradient_norm')]:metric_values[i].extend(c[key])
        metric_values[6].extend(c['layers']['output']['change_rms'])
    shared_limits={}
    for i,values in metric_values.items():
        arr=np.asarray(values,dtype=float);assert np.isfinite(arr).all()
        if i==1:shared_limits[i]=(min(-.1,float(arr.min())*1.1),max(1.1,float(arr.max())*1.1))
        else:
            assert np.all(arr>0)
            shared_limits[i]=(float(arr.min())/1.1,float(arr.max())*1.1)
    layer_limits={name:1.05*max(max(c['layers'][name]['change_rms'][52:103]) for c in r['curves'].values()) for name in NAMES}
    with PdfPages(target) as pdf:
        for group_start in range(0,len(ARMS),4):
            group=ARMS[group_start:group_start+4]
            fig, axes = plt.subplots(8, len(group), figsize=(max(12,6*len(group)), 16), sharex=True, sharey='row')
            handles = {}
            for j, intervention in enumerate(group):
                for arm in ['control', intervention]:
                    c = r['curves'][arm]
                    values = [c['loss'], c['eta_kappa_hat'], c['gradient_rho'], c['raw_gradient_norm'],
                              c['active_gradient_norm'], c['layers']['output']['rho'],
                              c['layers']['output']['change_rms'], c['lr']]
                    for i, y in enumerate(values):
                        line, = axes[i, j].plot(t, np.asarray(y, dtype=float), color=COLORS[arm], lw=1.15,
                            label=LABELS[arm], drawstyle='steps-post' if i == 7 else 'default')
                        if i == 0:
                            handles[arm] = line
                    axes[2, j].plot(t, c['active_gradient_rho'], color=COLORS[arm], ls='--', lw=.95)
                for i in [2, 5]:
                    axes[i, j].set_ylim(-1.05, 1.05)
                    axes[i, j].axhline(0, color='.5', lw=.6)
                for i in [0, 3, 4, 6]:
                    axes[i, j].set_yscale('log')
                axes[1, j].set_yscale('symlog', linthresh=1)
                for i in [0,1,3,4,6]:axes[i,j].set_ylim(*shared_limits[i])
                axes[1, j].axhline(1, color='.5', ls='--', lw=.7)
                axes[7, j].set_yscale('log')
                axes[7, j].set_ylim(.0005, .025)
                axes[7, j].set_yticks([.000625, .001, .002, .005, .01, .02], ['0.000625', '0.001', '0.002', '0.005', '0.01', '0.02'])
                for ax in axes[:, j]:
                    style(ax)
                    ax.set_xlim(448, 800)
                    ax.axvline(500, color='.4', ls=':', lw=.8)
                    ax.axvspan(500, 550, color='.5', alpha=.07, lw=0)
                axes[0, j].set_title(TITLE_BY_ARM[intervention], fontsize=12, pad=10)
                axes[-1, j].set_xlabel('Completed optimizer updates $t$')
            for i in range(8):
                # First data line is the identical baseline in every column.
                for j in range(1,len(group)):
                    np.testing.assert_array_equal(axes[i, 0].lines[0].get_ydata(), axes[i, j].lines[0].get_ydata())
            labels = ['(a) Training loss (log)', r'(b) $\eta\widehat{\kappa}_t$ (symlog)',
                      '(c) Gradient cosine', r'(d) Full $\|g_t\|$ (log)',
                      r'(e) Later-layer $\|g_t\|$ (log)', '(f) Output-change cosine',
                      '(g) Output RMS (log)', '(h) LR of later layers']
            for ax, label in zip(axes[:, 0], labels):
                ax.set_ylabel(label)
            legend(fig, [handles[k] for k in ['control', *group]], .957, group)
            fig.suptitle('MLP interventions versus the unchanged Adam baseline', fontsize=16, y=.987)
            fig.text(.5, .965, 'All interventions freeze first-linear weights + biases at 448; I2 onward additionally change LR at 500. Baseline remains unchanged.',
                     ha='center', fontsize=10)
            fig.text(.5, .025, r'$\eta\widehat{\kappa}_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$: projected-gradient diagnostic, not an Adam curvature threshold.'
                     '\nRow (c): solid = all 656,810 parameters; dashed = the same 42,210 later-layer parameters used in row (e).'
                     '\nSame 5,000 examples and axis scales. Output changes are outgoing. Shading: 500-550; dotted line: LR change. Log scales display the full dynamic range.',
                     ha='center', va='center', fontsize=9.5)
            fig.subplots_adjust(left=.065, right=.985, bottom=.072, top=.918, hspace=.23, wspace=.13)
            pdf.savefig(fig)
            plt.close(fig)

            # Keep the same intervention order in the layerwise column groups.
            fig, axes = plt.subplots(5, 2*len(group), figsize=(max(14,6.75*len(group)), 11), sharex=True)
            mask = (t >= 500) & (t <= 550)
            tt = t[mask]
            rowlabels = ['Linear 1 ($Z_1$)', 'Tanh 1 ($H_1$)', 'Linear 2 ($Z_2$)',
                         'Tanh 2 ($H_2$)', 'Linear 3 ($F$)']
            for j, intervention in enumerate(group):
                for i, (name, label) in enumerate(zip(NAMES, rowlabels)):
                    left, right = axes[i, 2*j:2*j+2]
                    for arm in ['control', intervention]:
                        c = r['curves'][arm]['layers'][name]
                        left.plot(tt, np.asarray(c['rho'], dtype=float)[mask], color=COLORS[arm], lw=1.15)
                        right.plot(tt, np.asarray(c['change_rms'])[mask], color=COLORS[arm], lw=1.15)
                    left.set_ylim(-1.05, 1.05)
                    left.axhline(0, color='.5', lw=.6)
                    if i < 2:
                        left.text(.5, .82, 'Intervention: zero motion\ncosine undefined', transform=left.transAxes,
                                  ha='center', fontsize=9, color=COLORS[intervention])
                    if j == 0:
                        left.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=12)
                    for ax in [left, right]:
                        style(ax)
                        ax.set_xlim(500, 550)
            for i in range(5):
                ymax = layer_limits[NAMES[i]]
                for k in range(1,2*len(group),2):
                    axes[i, k].set_yscale('symlog', linthresh=1e-4)
                    axes[i, k].set_ylim(bottom=-1e-5, top=ymax)
                for k in [0, 1]:
                    for j in range(2,2*len(group),2):
                        np.testing.assert_array_equal(axes[i, k].lines[0].get_ydata(), axes[i, k+j].lines[0].get_ydata())
            for ax, label in zip(axes[0], ['Change cosine', 'RMS (symlog)']*len(group)):
                ax.set_title(label, fontsize=11, pad=10)
            for ax in axes[-1]:
                ax.set_xlabel('Outgoing update $t$')
            legend(fig, [handles[k] for k in ['control', *group]], .955, group)
            fig.suptitle('Layerwise motion versus baseline: 500-550', fontsize=16, y=.987)
            fig.text(.5, .026, 'All 5,000 sample rows and every feature coordinate retained; matching scales per layer/metric. RMS: linear below 0.0001, logarithmic above.'
                     '\nThe first two representations are exactly fixed only in the intervention. Negative cosine does not alone imply a loss spike.',
                     ha='center', fontsize=10)
            fig.subplots_adjust(left=.065, right=.985, bottom=.09, top=.855, hspace=.27, wspace=.28)
            for j,title in enumerate([TITLE_BY_ARM[arm] for arm in group]):
                center=(axes[0,2*j].get_position().x0+axes[0,2*j+1].get_position().x1)/2
                fig.text(center, .900, title, ha='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    audit = dict(status='numeric_checks_passed', pages=2*((len(ARMS)+3)//4), pdf_sha256=sha(target), source_sha256=sha(source),
        generator_sha256=sha(Path(__file__)), full_window=[448, 800], layer_window=[500, 550], overview_rows=8,
        columns=[LABELS[arm]+' versus unchanged baseline' for arm in ARMS],
        scales={'loss_norms_output_rms':'log', 'eta_kappa_hat':'symlog, linear threshold1', 'layer_rms':'symlog, linear threshold0.0001'},
        maximum_interventions_per_page=4,shared_scales_across_pages=True,same_baseline_in_every_column=True, matched_axis_scales=True, zero_motion_cosines_undefined=True)
    (HERE/'figure_verification.json').write_text(json.dumps(audit, indent=2)+'\n', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
