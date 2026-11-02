# Observation002: requested smoothed Adam composite

## Scope and sources

The user requested one figure following the Analysis024 manuscript pattern,
then specified separate gradient/momentum norm rows and R_t only for Adam
updates. This is a display-only reanalysis of the same nine Run213 arms; no
training, model/data/optimizer change, new seed, Hessian work or cloud cost.
Earlier PDFs and executed scientific inputs remain unchanged.

The source cohort is fast_calibration_results.json. Each original manifest hash
and the bytes/SHA256 of its boundaries.jsonl and metrics.jsonl are checked by
09_plot_smoothed_composite.py. The source identities, every input/plotted value,
reference-PDF hashes and final PDF hash are in smoothed_composite_manifest.json.
The independent numerical and visual review is in smoothed_composite_verification.json.

## Caption

Full-batch Adam diagnostics over128 updates, with one column for each controlled
case: CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer. Rows are training
loss (log scale), raw-gradient eta*kappa_hat, consecutive cosine rho for raw
gradients, momentum and Adam directions, raw-gradient norm (log scale), stored
first-moment norm (log scale), and cumulative R_t for Adam's measured parameter
updates. Blue/orange/green identify the increasing constant learning rates listed
above each column. The two norm rows share the same vertical limits within each
column. The horizontal axis is Training step. Complete16-observation trailing
arithmetic means are shown in the first seven rows; cumulative R_t is unsmoothed.
The dashed line at eta*kappa_hat=2 matches the GD reference figure as a numerical
reference; it is not an identified Adam stability threshold.

## Definitions, reduction and timing

The raw-gradient expression is eta*kappa_hat = 1-rho*norm(g_current)/norm(g_previous),
computed pointwise before averaging. It is the same expression used in the GD
figures, evaluated here along Adam's trajectory. No Hessian is evaluated and no
GD directional-curvature identification is asserted for the adaptive step.

Momentum denotes actual stored first moments after consuming the corresponding
raw gradient, without bias correction. Adam rho uses the bias-corrected,
second-moment-normalized direction. R_t uses measured Adam parameter increments:
net displacement from initialization divided by their cumulative path length.
Only this Adam-update R_t is plotted. No raw-gradient or momentum R_t appears.

Loss covers parameter states0..128 (loss after that many completed updates).
Vector measurements are placed at their associated applied-update number1..128;
the first adjacent pair is at step2. Thus each raw gradient is the one consumed
by its labelled update, and its associated momentum/Adam direction uses it.
The separately measured terminal gradient is unapplied and excluded from all
vector rows. The first complete smoothed loss is at15; norm means start at16;
consecutive cosine and eta*kappa_hat means start at17. No initial partial window,
padding, extrapolation, logarithmic averaging or post-averaging reconstruction
is used. Log scaling is applied after the arithmetic means. R_t retains all128
original cumulative prefixes without smoothing or reset.

## Coverage, uncertainty and verification

Nine128-update conditions, one fixed initialization per case, three constant LRs
per case, no uncertainty bands or seed aggregation. Full training objectives,
parameter scope, FP32 training/FP64 diagnostics and Adam settings are unchanged.
Smoothing suppresses short fluctuations, delays/broadens transitions and omits
the incomplete initial window. Existing unsmoothed evidence remains available.
No stability, generalization, convergence or optimizer-benefit claim is added.

All72 curves /8,244 plotted values pass source and Matplotlib-artist checks.
Window means agree with independent convolution and scalar math.fsum reductions;
maximum absolute difference1.78e-15. All nine R_t traces equal their source arrays.
The one-page PDF contains24 panels; text bounds, manuscript notation, log scales,
shared norm scales and three Training step labels are checked. Final rendering
was visually reviewed with no clipped labels, overlaps or obscured curves.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/09_plot_smoothed_composite.py

Output: figures/adam_diagnostics_smoothed.pdf. The original manuscript figures
were used as visual references; manuscript text/assets were not edited.
