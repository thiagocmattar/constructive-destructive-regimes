# I6 and I5: separate two-column comparison

The new [Figure 02](../figures/02-i6-vs-i5-cifar10-mlp.pdf) places **I6 (LR 0.0025)
on the left and I5 (LR 0.005) on the right**, each against unchanged Adam.
Independent linear training-loss axes expose I6's delayed peak without I5's
larger excursion compressing it. Compare numerical ticks, not visual height,
across the loss panels. The other seven rows share scales across columns.

## Results and caption

Both interventions freeze first-linear weights and biases from state 448, then
change the remaining layers' Adam LR at state 500 (first changed update:
500 -> 501). The baseline updates all layers at LR 0.001 throughout. I5 and I6
branch from the same complete state 500, with no optimizer reset or unfreezing.

Over states 500-800, **I6 peaks at 0.0724478 at 707**, whereas **I5 peaks at
2.5857999 at 505**. The unchanged baseline peaks at 0.0550626 at 522. I6's
initially quiet trajectory therefore does not imply the absence of a later
excursion. These are the existing verified results; no new experiment or
diagnostic reduction was run for this figure.

**Caption.** Frozen-first-layer MLP interventions I6 and I5 versus unchanged
Adam on the same 5000 CIFAR-10 training examples. Rows show (a) full training
loss, (b) eta kappa hat, (c) consecutive raw-gradient cosine, (d) full gradient
norm, (e) later-layer gradient norm, (f) consecutive output-change cosine,
(g) output-change coordinate RMS, and (h) later-layer LR. Solid/dashed curves
in (c) denote full/later-layer gradients. Dotted lines mark state 500. Loss
panels use independent linear ranges; all other corresponding panels share
scales. Symmetric-log and logarithmic scales are labeled. All 353 states
448-800 are plotted without smoothing.

## Definitions and coverage

- Loss is the full 5000-example mean half squared error, not a smaller probe.
- Full raw gradients include derivatives of all 656810 parameters, including
  frozen parameters. Later-layer statistics use the same 42210-parameter
  subset in every arm. Gradients are not differenced.
- The projected-gradient diagnostic is
  $\eta\widehat\kappa_t=1-\langle g_t,g_{t-1}\rangle/\|g_{t-1}\|^2$.
  Under Adam, this is not an actual curvature estimate or stability threshold.
- Output changes are outgoing: $\Delta F_t=F_{t+1}-F_t$. Cosine compares
  $\Delta F_t$ with $\Delta F_{t-1}$ after flattening the entire 5000-by-10
  matrix. Coordinate RMS is $\|\Delta F_t\|_F/\sqrt{50000}$; there is no
  averaging across examples before measuring the motion. Retained state 801
  supplies the outgoing change at 800.

Matched data, architecture, optimizer and precision settings, native-run
qualification, source identities and original coverage are detailed in
[observation 003](003-lr-halving.md). This is one seed and one fork; there are
no uncertainty intervals or validation/generalization measurements. Lower
endpoint loss or smaller excursions alone do not establish better learning.

## Sources and reproduction

[05_i6_i5_figure.py](../05_i6_i5_figure.py) reads the baseline, `halve_0025` and
`halve_005` arrays from [comparison.json](../comparison.json), whose SHA-256 is
`31b98aa2a57baf58c2732d344c3bc2d697b7c3cbee26922bb19b67c164154bbe`.
The underlying interventions are Runs 249 and 248, respectively; the baseline
is the qualified retained control from Analysis 064. Run from the repository:

```powershell
python analyses/065-2026-11-07-mlp-frozen-lr-double/05_i6_i5_figure.py
```

[Numerical verification](../figure02_verification.json) checks all 12708
plotted values, identical baseline copies, independent loss axes and the
unchanged original Figure 01. [Visual verification](../figure02_visual_verification.json)
records the single-page render review. This new artifact leaves the existing
four-page overview intact. No new compute worker, storage allocation or cloud
charge was needed.
