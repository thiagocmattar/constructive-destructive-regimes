# Observation006: short-window training-loss differences, updates 1-400

The requested [figure variant](../figures/training_size_overlay_loss_diff_n5_1_400.pdf)
replaces the loss panel with a smoothed signed training-loss difference. Use a
short window N=5, chosen as the display setting. The other three panels retain
their unsmoothed raw values, solid model-size colours, log norm axis and horizontal
range 1..400. Runs017/018/019 and seed 1235 are reused without training.

For each model, first difference the raw minibatch training losses:

```text
delta_L[t] = L[t] - L[t-1],                       t = 2..400
mean_delta_L[t] = mean(delta_L[t-4], ..., delta_L[t])
                = (L[t] - L[t-5]) / 5,           t = 6..400.
```

Negative values indicate falling recorded loss; positive values indicate rising
loss. There is no absolute value, negation, relative normalization or second
smoothing pass. This is a complete trailing arithmetic mean of five adjacent
differences. The first five display points are undefined, with no shortened edge
windows or artificial zeros. Every mean through update 400 is shown.

Sources, matching and raw-gradient definitions follow
[observation005](005-unsmoothed-first400.md). In particular, successive losses
evaluate different minibatches, so the plotted difference is not a finite
difference of the same full objective. It is also not the norm of the gradient
or an AdamW update-direction statistic. One-seed coverage supplies no seed
uncertainty, causal mechanism or stability conclusion.

**Caption.** Pythia-14M/31M/70M MiniPile dynamics over updates 1-400, seed 1235.
The upper-left panel shows the complete five-update trailing mean of signed
adjacent training-loss differences, first defined at update 6. Negative means
indicate falling loss. The remaining panels retain unsmoothed eta*kappa_hat
gradient-change proxy, adjacent raw cosine rho and raw gradient L2 norm. Solid
blue/orange/green lines identify 14M/31M/70M. Only the norm axis is logarithmic;
the loss-change panel includes a zero reference. The first raw adjacent pair is
undefined. Earlier figures and data are preserved.

Generate with `14_plot_raw_first400.py --loss-diff`. The
[CSV](../training_size_overlay_loss_diff_n5_1_400_data.csv) retains raw loss and
diagnostics plus `loss_diff` and `loss_diff_mean`; undefined entries are blank.
The [manifest](../training_size_overlay_loss_diff_n5_1_400_manifest.json) verifies
source sizes/hashes, all 1,185 means against the independent telescoping formula,
all 4,779 finite plotted values and CSV roundtrip. The nine diagnostic curves are
checked to have exactly the same array hashes as observation005. Previous PDFs,
data and manifests are preserved byte for byte. After rendering with Poppler to
`scratch/training_size_overlay_loss_diff_n5_1_400` and inspecting the page, record
acceptance with `--loss-diff --accept-reviewed`. No new run/analysis ID is used.
