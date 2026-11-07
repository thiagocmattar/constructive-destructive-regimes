"""Signed leading coefficient of the existing fixed output-change SVD."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'runs/239-2026-11-07-mlp-output-change-alignment/artifacts/spark/payload/worker'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def summarize(states, alpha, lo, hi):
    a = alpha[(states >= lo) & (states <= hi)]
    product = a[1:] * a[:-1]
    weight = a[1:]**2 + a[:-1]**2
    return dict(range=[lo, hi], adjacent_pairs=len(product),
                sign_reversals=int((product < 0).sum()),
                zero_product_pairs=int((product == 0).sum()),
                reversal_fraction=float((product < 0).mean()),
                pair_energy_weighted_reversal_fraction=float(weight[product < 0].sum()/weight.sum()))


def main():
    torch.set_num_threads(4)
    audit = json.loads((HERE / 'output_heatmap_verification.json').read_text())
    pattern_path = HERE / audit['patterns']['path']
    assert sha(pattern_path) == audit['patterns']['sha256']
    for name in ['outputs.pt', 'boundaries.jsonl']:
        path = RAW / name
        assert sha(path) == audit['sources'][path.relative_to(ROOT).as_posix()]
    basis = torch.load(pattern_path, weights_only=False)
    saved = torch.load(RAW / 'outputs.pt', weights_only=False)
    states = np.asarray(basis['states'])
    np.testing.assert_array_equal(states, np.arange(448, 801))
    assert saved['states'] == list(range(447, 802))
    vector = basis['patterns'][0].numpy().reshape(-1)
    alpha = basis['coefficients'][:, 0].numpy()
    # Independent NumPy projection of raw outputs; no averaging or centering.
    outputs = saved['outputs'].numpy().astype(np.float64).reshape(355, -1)
    delta = outputs[2:] - outputs[1:-1]
    direct = delta @ vector
    np.testing.assert_allclose(alpha, direct, rtol=0, atol=1e-10)
    assert abs(vector @ vector - 1) < 1e-12
    # An increment coefficient must equal the difference of projected positions.
    position = (outputs[1:] - outputs[1]) @ vector
    np.testing.assert_allclose(np.diff(position), alpha, rtol=0, atol=1e-10)
    with (HERE / 'svd_output_heatmap_values.csv').open() as f:
        previous = list(csv.DictReader(f))
    np.testing.assert_allclose(alpha / np.sqrt(5000),
        [float(r['mode_01_coefficient_over_sqrtN']) for r in previous], rtol=0, atol=1e-14)
    boundaries = {r['state']: r for r in map(json.loads, (RAW / 'boundaries.jsonl').read_text().splitlines())}
    loss = np.array([boundaries[int(t)]['loss'] for t in states])
    summaries = [summarize(states, alpha, lo, hi) for lo, hi in
                 [(448, 800), (465, 474), (512, 522), (465, 485), (512, 529)]]
    energy = float((alpha**2).sum() / (delta**2).sum())
    assert abs(energy - float(basis['energy_fractions'][0])) < 1e-12
    values = HERE / 'leading_svd_coefficient_values.csv'
    with values.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['state', 'loss', 'alpha', 'projected_position_relative_to_state448', 'reversal_from_previous_alpha'])
        for i, t in enumerate(states):
            writer.writerow([int(t), loss[i], alpha[i], position[i],
                             '' if i == 0 else int(alpha[i]*alpha[i-1] < 0)])

    plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'font.size': 11, 'axes.titlesize': 12})
    fig = plt.figure(figsize=(11.5, 10))
    grid = fig.add_gridspec(3, 2, left=.09, right=.97, bottom=.12, top=.89,
                          hspace=.50, wspace=.24, height_ratios=[1, 1.3, 1.15])
    ax_loss = fig.add_subplot(grid[0, :])
    ax_alpha = fig.add_subplot(grid[1, :], sharex=ax_loss)
    ax_loss.plot(states, loss, color='#2166ac', lw=1.2)
    ax_loss.set(title='(a) Full training loss', ylabel=r'$L_t$')
    ax_loss.tick_params(labelbottom=False)
    ax_alpha.plot(states, alpha, color='#555555', lw=.8)
    ax_alpha.scatter(states, alpha, c=np.where(alpha >= 0, '#b2182b', '#2166ac'), s=7, zorder=3)
    ax_alpha.set(title='(b) Leading signed coefficient: 306 / 352 adjacent pairs reverse sign (86.9%)',
                 ylabel=r'$\alpha_t=\langle F_{t+1}-F_t,v_1\rangle$', xlabel='Completed optimizer updates, t')
    ax_alpha.set_xlim(448, 800)
    ax_alpha.set_xticks([448, 512, 576, 640, 704, 768, 800])
    axes = [ax_loss, ax_alpha]
    for j, (lo, hi, peak) in enumerate([(465, 485, 474), (512, 529, 522)]):
        ax = fig.add_subplot(grid[2, j])
        mask = (states >= lo) & (states <= hi)
        ax.plot(states[mask], alpha[mask], color='#555555', lw=1)
        ax.scatter(states[mask], alpha[mask], c=np.where(alpha[mask] >= 0, '#b2182b', '#2166ac'), s=24, zorder=3)
        ax.axvline(peak, color='#999999', ls='--', lw=.8, label=f'Loss peak: {peak}')
        ax.set(title=f'({chr(99+j)}) Zoom: states {lo}-{hi}', xlabel='Completed optimizer updates, t', ylabel=r'$\alpha_t$')
        ax.legend(frameon=False, fontsize=9, loc='lower left')
        ax.set_xticks(np.arange(lo, hi+1, 3))
        axes.append(ax)
        for target in [ax_loss, ax_alpha]:
            target.axvspan(lo, hi, color='#999999', alpha=.10)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15, lw=.6)
        if ax is not ax_loss:
            ax.axhline(0, color='gray', lw=.7, ls=':')
    fig.suptitle('CIFAR-10 MLP: back-and-forth motion along the leading functional axis', fontsize=16, y=.97)
    fig.text(.53, .935, 'Unchanged Adam control | Fixed SVD of output differences | All 5,000 examples and 10 outputs retained', ha='center', fontsize=11)
    fig.text(.53, .065, r'$v_1$ is one fixed unit vector in $\mathbb{R}^{50000}$; it captures 25.17% of full-window squared change energy.', ha='center')
    fig.text(.53, .042, r'Raw coefficients shown (heatmap row 1 multiplied by $\sqrt{5000}$). Positive and negative mean opposite motion.', ha='center', fontsize=10)
    fig.text(.53, .020, r'$\alpha_t$ describes update $t\to t+1$; $L_t$ is the loss before that update. Basis fitted over the entire window.', ha='center', fontsize=10)
    pdf = HERE / 'figures/06-leading-svd-coefficient.pdf'
    fig.savefig(pdf)
    plt.close(fig)
    result = dict(status='numeric_checks_passed', state_range=[448,800],
        definition='alpha_t=<flatten(F_(t+1)-F_t),v_1>; fixed unit v_1 from existing uncentered full-window SVD',
        energy_fraction=energy, summaries=summaries,
        pair_weight='alpha_t^2+alpha_(t-1)^2; not a fraction of total network displacement energy',
        projection_max_absolute_error=float(np.max(abs(alpha-direct))),
        position_difference_max_absolute_error=float(np.max(abs(np.diff(position)-alpha))),
        source_sha256={str(p.relative_to(ROOT).as_posix()): sha(p) for p in
            [RAW/'outputs.pt', RAW/'boundaries.jsonl', pattern_path, HERE/'svd_output_heatmap_values.csv']},
        generator_sha256=sha(Path(__file__)), csv_sha256=sha(values), pdf_sha256=sha(pdf))
    (HERE / 'leading_svd_coefficient_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
