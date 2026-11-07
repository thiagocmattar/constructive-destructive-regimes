"""A separate two-column I6/I5 overview with independent linear loss ranges."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ['halve_0025', 'halve_005']
COLORS = {'control': '#2166ac', 'halve_0025': '#a64d79', 'halve_005': '#b8860b'}
LABELS = {'control': 'Baseline: unchanged Adam', 'halve_0025': 'I6: freeze + LR 0.0025',
          'halve_005': 'I5: freeze + LR 0.005'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                    'font.size': 10, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def values(curve):
    return [curve['loss'], curve['eta_kappa_hat'], curve['gradient_rho'], curve['raw_gradient_norm'],
            curve['active_gradient_norm'], curve['layers']['output']['rho'],
            curve['layers']['output']['change_rms'], curve['lr']]


def main():
    source = HERE/'comparison.json'
    record = json.loads(source.read_text())
    prior_audit = json.loads((HERE/'figure_verification.json').read_text())
    assert record['status'] == 'verified' and sha(source) == prior_audit['source_sha256']
    assert record['generator_sha256'] == sha(HERE/'01_reduce.py')
    original = HERE/'figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf'
    original_sha = sha(original)
    assert original_sha == prior_audit['pdf_sha256']
    t = np.array(record['states'])
    assert np.array_equal(t, np.arange(448, 801))
    selected = {arm: record['curves'][arm] for arm in ['control', *ARMS]}
    fig, axes = plt.subplots(8, 2, figsize=(12, 16), sharex=True)
    for i in range(1, 8):
        axes[i, 1].sharey(axes[i, 0])
        axes[i, 1].tick_params(labelleft=False)
    handles = {}
    plotted = 0
    loss_limits = {}
    for j, intervention in enumerate(ARMS):
        for arm in ['control', intervention]:
            curve = selected[arm]
            for i, y in enumerate(values(curve)):
                data = np.asarray(y, dtype=float)
                assert data.shape == t.shape and np.isfinite(data).all()
                line, = axes[i, j].plot(t, data, color=COLORS[arm], lw=1.2,
                    drawstyle='steps-post' if i == 7 else 'default', label=LABELS[arm])
                np.testing.assert_array_equal(line.get_ydata(), data)
                plotted += len(data)
                if i == 0:
                    handles[arm] = line
            data = np.asarray(curve['active_gradient_rho'], dtype=float)
            line, = axes[2, j].plot(t, data, color=COLORS[arm], ls='--', lw=1)
            np.testing.assert_array_equal(line.get_ydata(), data)
            plotted += len(data)
        # Independent loss ranges, based only on the two curves in this column.
        loss_max = max(max(selected[arm]['loss']) for arm in ['control', intervention])
        axes[0, j].set_ylim(0, loss_max*1.09)
        loss_limits[intervention] = list(axes[0, j].get_ylim())
        axes[0, j].tick_params(labelleft=True)
        loss = np.asarray(selected[intervention]['loss'])
        peak = int(np.argmax(loss))
        axes[0, j].annotate(f'{loss[peak]:.5g} at {t[peak]}', xy=(t[peak], loss[peak]),
                           xytext=(12, -16), textcoords='offset points', fontsize=9,
                           color=COLORS[intervention], ha='left', va='top',
                           arrowprops={'arrowstyle': '-', 'color': COLORS[intervention], 'lw': .7})
        for i in [2, 5]:
            axes[i, j].set_ylim(-1.05, 1.05)
            axes[i, j].axhline(0, color='.5', lw=.6)
        for i in [3, 4, 6]:
            axes[i, j].set_yscale('log')
        axes[1, j].set_yscale('symlog', linthresh=1)
        axes[1, j].axhline(1, color='.5', ls='--', lw=.7)
        axes[7, j].set_ylim(.0008, .0054)
        axes[7, j].set_yticks([.001, .0025, .005], ['0.001', '0.0025', '0.005'])
        for ax in axes[:, j]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.2, lw=.6)
            ax.set_xlim(448, 800)
            ax.axvline(500, color='.4', ls=':', lw=.8)
        axes[0, j].set_title(LABELS[intervention] + ' from 500', fontsize=13, pad=10)
        axes[-1, j].set_xlabel('Completed optimizer updates $t$')
    for i in [1, 3, 4, 6]:
        all_values = np.concatenate([np.asarray(values(c)[i]) for c in selected.values()])
        if i == 1:
            axes[i, 0].set_ylim(min(-.1, float(all_values.min())*1.1),
                                max(1.1, float(all_values.max())*1.1))
        else:
            assert np.all(all_values > 0)
            axes[i, 0].set_ylim(float(all_values.min())/1.1, float(all_values.max())*1.1)
    for i in range(8):
        np.testing.assert_array_equal(axes[i, 0].lines[0].get_ydata(), axes[i, 1].lines[0].get_ydata())
    assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[0, 1])
    assert loss_limits[ARMS[0]] != loss_limits[ARMS[1]]
    rowlabels = ['(a) Full training loss', r'(b) $\eta\widehat{\kappa}_t$ (symlog)',
                 '(c) Gradient cosine', r'(d) Full $\|g_t\|$ (log)',
                 r'(e) Later-layer $\|g_t\|$ (log)', '(f) Output-change cosine',
                 '(g) Output RMS (log)', '(h) LR of later layers']
    for ax, label in zip(axes[:, 0], rowlabels):
        ax.set_ylabel(label)
    fig.suptitle('MLP: I6 and I5 versus unchanged Adam', fontsize=16, y=.987)
    fig.text(.5, .967, 'First-linear weights + biases frozen at 448; learning rate changes at 500; same 5,000 examples.',
             ha='center', fontsize=10)
    fig.legend([handles[k] for k in ['control', *ARMS]], [LABELS[k] for k in ['control', *ARMS]],
               loc='upper center', bbox_to_anchor=(.55, .956), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .024, 'Training loss: independent linear ranges. Other rows share scales; dotted line: LR change at 500.'
             '\nRow (c): solid = full raw gradient (656,810 parameters); dashed = later layers (42,210), also used in row (e).'
             '\n' + r'$\eta\widehat{\kappa}_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$ is a projected-gradient diagnostic under Adam. Output changes are outgoing.',
             ha='center', va='center', fontsize=9)
    fig.subplots_adjust(left=.105, right=.98, bottom=.07, top=.919, hspace=.23, wspace=.21)
    target = HERE/'figures/02-i6-vs-i5-cifar10-mlp.pdf'
    fig.savefig(target)
    plt.close(fig)
    assert sha(original) == original_sha
    audit = dict(status='numeric_checks_passed', pages=1, column_order=['I6', 'I5'], window=[448, 800],
                 source_sha256=sha(source), generator_sha256=sha(Path(__file__)), pdf_sha256=sha(target),
                 plotted_values_checked=plotted, baseline_identical_in_both_columns=True,
                 training_loss_scale='linear, independent per column', training_loss_limits=loss_limits,
                 other_rows_share_scales=True, original_figure_sha256=original_sha, original_figure_unchanged=True)
    (HERE/'figure02_verification.json').write_text(json.dumps(audit, indent=2)+'\n', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
