# Observation003: loss, raw cumulative ratio, smoothed alignment and vector norms

Question: place training loss and raw-gradient B/A beside adjacent raw/momentum cosine and their individual norms, using the existing seven-LR color palette and a longer raw-cosine rolling average. The six requested panels are labeled sequentially(a)-(f), resolving the duplicated(b) in the dictated request. This is a new display of the existing trajectories, not a new experiment or empirical hypothesis.

[Figure](../figures/loss_ratio_alignment_norms.pdf), [exact plotted series, smoothing contract and hashes](../loss_ratio_alignment_norms_data.json), [PDF/render and aggregation verification](../verification.json). Generator: `../../11_plot_loss_ratio_alignment.py`. Shared source/color/legend helpers: `../../10_plot_lr_geometry.py`; loss averaging and vector extraction use the unchanged Analysis008 `08_plot_simplified.py` helpers.

## Sources and coverage

Source: unchanged `../comparison_data.json`, SHA256 a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739, containing the verified event measurements and original source hashes for Runs021-027. Each full pass has712 accepted updates and covers1,491,711,416 input tokens,1,490,983,041 supervised targets and728375 packed sequences. Display includes full-sized updates1-711. The partial final update712 remains in source data and numerical results. No measurement or training input was changed.

## Panels, reduction and caption

Row-major3x2 layout:

- (a) Training loss: trailing25-update token-weighted arithmetic mean of token-mean cross-entropy in nats; light traces show individual update losses.
- (b) Raw-gradientg B/A: unsmoothed recorded cumulative ratio, B=norm of sum and A=sum of norms.
- (c) Raw-gradient adjacent cosine: trailing50-update arithmetic mean, replacing the shorter25-update median used in earlier metric views. Individual cosine traces are omitted to keep this panel smooth.
- (d) Momentum adjacent cosine: trailing25-update arithmetic mean; individual cosine traces omitted.
- (e) Raw-gradient individual vector norm: trailing10-update arithmetic mean of the norms, computed before plotting on logarithmic Y. All displayed norms are strictly positive; no point is discarded for log display. This is the mean of scalar norms, not the norm of an averaged vector.
- (f) Momentum individual vector norm: unsmoothed recorded norm on linear Y.

All smoothing windows include the current update and the previouswindow-1 updates, without future data. At the beginning, the window uses available prefix measurements. The undefined first adjacent cosine is omitted from the average; its own displayed point remains a gap. Window placement and arithmetic averaging were checked against the source measurements at update400 for every LR.

Caption/legend: colors match Observation002 exactly (`plasma_r` over normalizedlog10 LR, colormap0.10-0.95), from light yellow at the smallest LR to dark purple at the largest. All seven LRs appear in every panel. Dashed curves identify original Spark runs(S); solid curves identify new RTX4090 runs(R). g is pre-clipping; m is actual uncorrected Adam exp_avg from clipped gradients, captured before preconditioning. Seed1234, frozen initial tensors and token order; LR scales AdamW decay.

## Uncertainty and interpretation limits

One seed and a finite full-pass budget; no seed-level uncertainty band. Hardware/runtime differ across original/new groups. Moving averages attenuate spikes and can delay displayed zero crossings: the smoothed sign is not the first negative raw pair or a convergence/stability threshold. Norm smoothing does not measure gradient noise variance or per-sample alignment. Adjacent cosine does not identify long-lag destructive interference. The figure introduces no new causal or performance claim; the completed-run results in Observation001 are unchanged. Previous PDFs are retained.
