# Observation009: N=5 and N=21 smoothing for every metric

The user requested smoothing for all four rows of the split-range figure:
N=5 for updates 1-100 and N=21 for 100-400. The
[revised figure](../figures/training_size_overlay_log_loss_diff_split_n5_n21.pdf)
keeps the metric-row order, solid model-size colours, independent column Y limits
and logarithmic norm axes. Earlier figures, data and manifests are preserved.
No training, new seed or new run/analysis ID is introduced.

Source trajectories and definitions follow
[observation007](007-log-loss-difference.md): Pythia-14M/31M/70M,
Runs017/018/019, seed 1235, and matched data/order/AdamW recipe. The generator
reads the verified CSV's RAW `log_loss_diff`, `eta_kappa_proxy`, `rho` and
`raw_gradient_norm` columns. It does not average the prior N=5 loss-change means.

For each column's N, compute complete trailing arithmetic means on the full
history before selecting the visible update range:

```text
M_N[x][t] = mean(x[t-N+1], ..., x[t])
loss row: x[t] = ln(L[t]) - ln(L[t-1])
          M_N[x][t] = ln(L[t]/L[t-N]) / N
other rows: x[t] is the raw per-update q, rho or norm scalar.
```

The first complete left-column means are at update 5 for norms and 6 for loss,
q and rho (whose first raw pair is undefined). There are 96/95/95/95 finite
left-column values per model in norm/loss/q/rho order. The right column has 301
finite values for every metric/model. No shortened windows, interpolation or
artificial zeros are used. Negative log-loss change means falling recorded loss.

Update 100 appears in both columns with DIFFERENT smoothing windows: N=5 uses
raw scalar updates 96..100, while N=21 uses 80..100. For loss differences these
correspond to endpoint losses 95/100 and 79/100, respectively. Smoothing is not
reset at the column boundary. The figure has 4,755 finite plotted means. The
1,203 CSV rows include each model's step 100 twice, keyed by column and window.

Norms are averaged arithmetically before the log-axis transform: this is mean
norm, not the norm of a mean gradient or root mean square norm. q is averaged
after its raw per-update calculation, not computed from smoothed rho/norms.
Smoothed q>1 and smoothed rho<0 need not be exactly equivalent. Different window
lengths can shift or blur apparent transitions, so exact event timing comes from
the raw histories. Different minibatches, AdamW and one-seed coverage retain
the prior limits on same-objective improvement, curvature, stability and causal
model-size interpretations. This display change adds no scientific finding.

**Caption.** Pythia MiniPile dynamics with column-specific smoothing, seed 1235.
Columns show updates 1-100 with N=5 and 100-400 with N=21. Every row uses a complete
trailing arithmetic mean of its underlying per-update values. Rows show adjacent
natural-log loss change, eta*kappa_hat gradient-change proxy, adjacent raw-gradient
cosine rho and raw gradient L2 norm. Solid blue/orange/green curves identify
14M/31M/70M. Y limits vary by column; norm means use log Y. Update 100 appears
with both windows, whose histories span the visible boundary.

Generate with `15_plot_split_windows.py --smooth-all`. The
[plot CSV](../training_size_overlay_log_loss_diff_split_n5_n21_data.csv) records
size, column, N, update and all plotted means; undefined values are blank. The
[manifest](../training_size_overlay_log_loss_diff_split_n5_n21_manifest.json)
records source/output hashes, every independent sliced-window check, telescoping
log-loss checks, all 4,755 plotted-array checks, CSV roundtrip and preserved-file
hashes. Render with Poppler to `scratch/training_size_overlay_log_loss_diff_split_n5_n21`,
inspect the page, then record acceptance with `--smooth-all --accept-reviewed`.
