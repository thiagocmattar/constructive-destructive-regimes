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
COLORS = {'control': '#2166ac', 'freeze': '#d95f02', 'double': '#7b3294', 'high': '#c51b2f'}
LABELS = {'control': 'Baseline: unchanged Adam', 'freeze': 'I1: freeze linear 1',
          'double': 'I2: freeze + LR 0.002 at 500', 'high': 'I3: freeze + LR 0.02 at 500'}
ARMS = ['freeze', 'double', 'high']
TITLES = ['I1: freeze linear 1; LR 0.001', 'I2: freeze + LR 0.002 from 500', 'I3: freeze + LR 0.02 from 500']
NAMES = ['preactivation1', 'hidden1', 'preactivation2', 'hidden2', 'output']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2, lw=.6)


def legend(fig, handles, y):
    fig.legend(handles, [LABELS[k] for k in ['control', *ARMS]],
               loc='upper center', bbox_to_anchor=(.53, y), ncol=4, frameon=False, fontsize=10)


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
    with PdfPages(target) as pdf:
        fig, axes = plt.subplots(8, 3, figsize=(18, 16), sharex=True, sharey='row')
        handles = {}
        for j, intervention in enumerate(ARMS):
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
            axes[1, j].axhline(1, color='.5', ls='--', lw=.7)
            axes[7, j].set_yscale('log')
            axes[7, j].set_ylim(.0008, .025)
            axes[7, j].set_yticks([.001, .002, .02], ['0.001', '0.002', '0.02'])
            for ax in axes[:, j]:
                style(ax)
                ax.set_xlim(448, 800)
                ax.axvline(500, color='.4', ls=':', lw=.8)
                ax.axvspan(500, 550, color='.5', alpha=.07, lw=0)
            axes[0, j].set_title(TITLES[j], fontsize=12, pad=10)
            axes[-1, j].set_xlabel('Completed optimizer updates $t$')
        for i in range(8):
            # First data line is the identical baseline in both columns.
            for j in [1, 2]:
                np.testing.assert_array_equal(axes[i, 0].lines[0].get_ydata(), axes[i, j].lines[0].get_ydata())
        labels = ['(a) Training loss (log)', r'(b) $\eta\widehat{\kappa}_t$ (symlog)',
                  '(c) Gradient cosine', r'(d) Full $\|g_t\|$ (log)',
                  r'(e) Later-layer $\|g_t\|$ (log)', '(f) Output-change cosine',
                  '(g) Output RMS (log)', '(h) LR of later layers']
        for ax, label in zip(axes[:, 0], labels):
            ax.set_ylabel(label)
        legend(fig, [handles[k] for k in ['control', *ARMS]], .957)
        fig.suptitle('MLP interventions versus the unchanged Adam baseline', fontsize=16, y=.987)
        fig.text(.5, .965, 'All interventions freeze first-linear weights + biases at 448; I2/I3 additionally change LR at 500. Baseline remains unchanged.',
                 ha='center', fontsize=10)
        fig.text(.5, .025, r'$\eta\widehat{\kappa}_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$: projected-gradient diagnostic, not an Adam curvature threshold.'
                 '\nRow (c): solid = all 656,810 parameters; dashed = the same 42,210 later-layer parameters used in row (e).'
                 '\nSame 5,000 examples and axis scales. Output changes are outgoing. Shading: 500-550; dotted line: I2/I3 LR change. Log scales display the full dynamic range.',
                 ha='center', va='center', fontsize=9.5)
        fig.subplots_adjust(left=.085, right=.985, bottom=.072, top=.918, hspace=.23, wspace=.13)
        pdf.savefig(fig)
        plt.close(fig)

        # Same interventions remain the three column groups on the layerwise page.
        fig, axes = plt.subplots(5, 6, figsize=(20, 11), sharex=True,
                                 gridspec_kw={'width_ratios': [1]*6})
        mask = (t >= 500) & (t <= 550)
        tt = t[mask]
        rowlabels = ['Linear 1 ($Z_1$)', 'Tanh 1 ($H_1$)', 'Linear 2 ($Z_2$)',
                     'Tanh 2 ($H_2$)', 'Linear 3 ($F$)']
        for j, intervention in enumerate(ARMS):
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
            ymax = max(axes[i, k].get_ylim()[1] for k in [1, 3, 5])
            for k in [1, 3, 5]:
                axes[i, k].set_yscale('symlog', linthresh=1e-4)
                axes[i, k].set_ylim(bottom=-1e-5, top=ymax)
            for k in [0, 1]:
                for j in [2, 4]:
                    np.testing.assert_array_equal(axes[i, k].lines[0].get_ydata(), axes[i, k+j].lines[0].get_ydata())
        for ax, label in zip(axes[0], ['Change cosine', 'RMS (symlog)']*3):
            ax.set_title(label, fontsize=11, pad=10)
        for ax in axes[-1]:
            ax.set_xlabel('Outgoing update $t$')
        legend(fig, [handles[k] for k in ['control', *ARMS]], .955)
        fig.suptitle('Layerwise motion versus baseline: 500-550', fontsize=16, y=.987)
        for center, title in zip([.23, .54, .85], TITLES):
            fig.text(center, .900, title, ha='center', fontsize=12)
        fig.text(.5, .026, 'All 5,000 sample rows and every feature coordinate retained; matching scales per layer/metric. RMS: linear below 0.0001, logarithmic above.'
                 '\nThe first two representations are exactly fixed only in the intervention. Negative cosine does not alone imply a loss spike.',
                 ha='center', fontsize=10)
        fig.subplots_adjust(left=.09, right=.985, bottom=.09, top=.855, hspace=.27, wspace=.28)
        pdf.savefig(fig)
        plt.close(fig)
    audit = dict(status='numeric_checks_passed', pages=2, pdf_sha256=sha(target), source_sha256=sha(source),
        generator_sha256=sha(Path(__file__)), full_window=[448, 800], layer_window=[500, 550], overview_rows=8,
        columns=['freeze versus unchanged baseline', 'freeze + LR0.002 versus unchanged baseline', 'freeze + LR0.02 versus unchanged baseline'],
        scales={'loss_norms_output_rms':'log', 'eta_kappa_hat':'symlog, linear threshold1', 'layer_rms':'symlog, linear threshold0.0001'},
        same_baseline_in_every_column=True, matched_axis_scales=True, zero_motion_cosines_undefined=True)
    (HERE/'figure_verification.json').write_text(json.dumps(audit, indent=2)+'\n', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
