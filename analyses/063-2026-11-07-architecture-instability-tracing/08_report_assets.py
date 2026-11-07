"""Compact vector figures and numerical LaTeX tables for the technical report."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'report'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'lines.linewidth': 1.0})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style(ax, t, peak):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18, linewidth=.5)
    ax.axvline(peak, color='.45', ls=':', lw=.7, zorder=0)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(length=2, pad=2)


def main():
    assets = REPORT / 'figures'
    assets.mkdir(parents=True, exist_ok=True)
    cases = ['mlp', 'resnet', 'transformer']
    paths = [HERE / (('mlp_450_500' if c == 'mlp' else c) + '_results.json') for c in cases]
    data = {c: json.loads(p.read_text()) for c, p in zip(cases, paths)}
    names = {'mlp': ['preactivation1', 'hidden2', 'output'],
        'resnet': ['conv1.call0.output', 'layer3.4.relu.call1.output', 'output.log_probabilities'],
        'transformer': ['encoder.call0.output', 'transformer_encoder.layers.1.norm2.call0.output', 'output.log_probabilities']}
    labels = {'mlp': ['Linear 1', 'Hidden 2', 'Output'],
        'resnet': ['First conv.', 'Last residual', 'Output'],
        'transformer': ['Embedding', 'Last block', 'Output']}
    titles = ['MLP / CIFAR-10', 'ResNet-32 / CIFAR-10', 'Transformer / WikiText-2']
    colors = ['#2166ac', '#7b3294', '#b2182b']
    fig, axes = plt.subplots(4, 3, figsize=(7.05, 7.35), sharex='col')
    peaks = {}
    for j, case in enumerate(cases):
        result = data[case]
        t = np.array(result['states'])
        lookup = {q['state']: q for q in result['boundaries']}
        loss = np.array([lookup[int(s)]['loss'] for s in t])
        ploss = np.array([lookup[int(s)]['probe_loss'] for s in t])
        peak = int(t[np.argmax(loss)])
        peaks[case] = peak
        curves = result['curves']
        axes[0, j].plot(t, loss, color='black', label='Full = probe' if case == 'mlp' else 'Full objective')
        if case != 'mlp':
            axes[0, j].plot(t, ploss, color='#b15928', ls='--', label='Fixed probe')
        else:
            np.testing.assert_array_equal(loss, ploss)
        axes[0, j].set_title(f'{titles[j]}\n$t={t[0]}$ to ${t[-1]}$', pad=7)
        axes[0, j].legend(frameon=False, loc='upper left' if case == 'mlp' else 'upper right', handlelength=1.4)
        axes[1, j].plot(t, curves['raw_gradient']['rho'], color=colors[0], label='Raw gradient')
        axes[1, j].plot(t, curves[names[case][-1]]['rho'], color=colors[2], label='Output changes')
        axes[1, j].legend(frameon=False, loc='upper left', handlelength=1.4)
        for name, label, color in zip(names[case], labels[case], colors):
            axes[2, j].plot(t, curves[name]['rho'], color=color, label=label)
            rms = np.array(curves[name]['change_rms'])
            axes[3, j].plot(t, rms / rms.max(), color=color)
        axes[2, j].legend(frameon=False, loc='lower right' if case == 'transformer' else 'upper left', ncol=1, handlelength=1.4)
        for i in [1, 2]:
            axes[i, j].set_ylim(-1.05, 1.05)
            axes[i, j].axhline(0, color='.5', lw=.5)
        axes[3, j].set_ylim(0, 1.05)
        axes[3, j].set_xlabel('Completed updates $t$')
        for ax in axes[:, j]:
            style(ax, t, peak)
    for ax, label in zip(axes[:, 0], ['(a) Objective loss', '(b) Consecutive cosine',
                                     '(c) Layer-change cosine', '(d) RMS / own maximum']):
        ax.set_ylabel(label, labelpad=6)
    fig.subplots_adjust(left=.095, right=.995, top=.92, bottom=.065, hspace=.20, wspace=.32)
    fig.savefig(assets / 'architecture-comparison.pdf')
    plt.close(fig)

    result = data['mlp']
    t = np.array(result['states'])
    stages = [s for s in result['stages'] if s['name'] != 'raw_gradient']
    stage_labels = ['$Z_1$: linear 1', '$H_1$: tanh 1', '$Z_2$: linear 2', '$H_2$: tanh 2', '$F$: linear output']
    fig, axes = plt.subplots(5, 3, figsize=(7.05, 6.10), sharex='col')
    for i, (stage, label) in enumerate(zip(stages, stage_labels)):
        curve = result['curves'][stage['name']]
        axes[i, 0].plot(t, curve['rho'], color='#2166ac')
        axes[i, 0].axhline(0, color='.5', lw=.5)
        axes[i, 0].set_ylim(-1.05, .15)
        axes[i, 0].set_ylabel(label, fontsize=9)
        axes[i, 1].plot(t, curve['change_rms'], color='#542788')
        axes[i, 1].set_ylim(bottom=0)
        axes[i, 2].plot(t, np.array(curve['alpha']) / np.sqrt(stage['coordinates']), color='#b2182b')
        axes[i, 2].axhline(0, color='.5', lw=.5)
        for ax in axes[i]:
            style(ax, t, peaks['mlp'])
    for ax, label in zip(axes[0], ['Consecutive cosine $\\rho_t$', 'Displacement RMS $r_t$', '$\\alpha_{t,1}/\\sqrt{P}$ (fixed basis)']):
        ax.set_title(label, fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel('Completed updates $t$')
    fig.subplots_adjust(left=.105, right=.995, top=.94, bottom=.075, hspace=.22, wspace=.34)
    fig.savefig(assets / 'mlp-layer-detail.pdf')
    plt.close(fig)

    records = {c: {s['name']: s for s in data[c]['stages']} for c in cases}
    rows = []
    def row(label, vals):
        rows.append(label + ' & ' + ' & '.join(str(v) for v in vals) + r' \\')
    row('Updates in fitted window ($T$)', [len(data[c]['states']) for c in cases])
    row('First learned stage: negative $\\rho_t$', [f"{records[c][names[c][0]]['negative']}/{len(data[c]['states'])}" for c in cases])
    row('Output: negative $\\rho_t$', [f"{records[c][names[c][-1]]['negative']}/{len(data[c]['states'])}" for c in cases])
    rows.append(r'\midrule')
    for key, label in [('top1', 'Output energy: first mode'), ('top10', 'Output energy: first ten modes')]:
        row(label, [f"{100*records[c][names[c][-1]][key]:.2f}\\%" for c in cases])
    row('Output modes for 95\\% energy', [records[c][names[c][-1]]['rank95'] for c in cases])
    row('Raw-gradient energy: first ten modes', [f"{100*records[c]['raw_gradient']['top10']:.2f}\\%" for c in cases])
    rows.append(r'\midrule')
    row('$\\operatorname{corr}(\\rho^g,\\rho^F)$', [f"{records[c]['raw_gradient']['correlation_with_output']:.3f}" for c in cases])
    for idx, label in [(0, 'First learned stage'), (1, 'Last hidden stage')]:
        row(label + r': $\operatorname{corr}(\rho^\ell,\rho^F)$', [f"{records[c][names[c][idx]]['correlation_with_output']:.3f}" for c in cases])
        row(label + r': $O_{10}(\ell,F)$', [f"{records[c][names[c][idx]]['temporal_top10_overlap']:.3f}" for c in cases])
    table = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Diagnostic & MLP & ResNet & Transformer \\', r'\midrule',
             *rows, r'\bottomrule', r'\end{tabular}']
    (REPORT / 'results-table.tex').write_text('\n'.join(table) + '\n', newline='\n')
    artifacts = [assets / 'architecture-comparison.pdf', assets / 'mlp-layer-detail.pdf', REPORT / 'results-table.tex']
    audit = dict(status='numeric_checks_passed', sources={p.name: sha(p) for p in paths},
        generator_sha256=sha(Path(__file__)), artifacts={p.relative_to(HERE).as_posix(): sha(p) for p in artifacts},
        plotted_windows={c: [data[c]['states'][0], data[c]['states'][-1]] for c in cases},
        peaks=peaks, mlp_svd_refitted=True, plot_reduction='No smoothing or sample averaging; RMS scaled only in overview row d')
    (HERE / 'report_assets_verification.json').write_text(json.dumps(audit, indent=2) + '\n', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
