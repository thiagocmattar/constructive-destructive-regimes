# Observation001: short full-objective Adam calibration

## Sources and reduction

Run213 owns nine independent 128-update arms under artifacts/attempt-01.
The source manifest for every arm, exact config/input identities, final artifact
bytes/SHA256, labels and continuous summaries are listed in
../fast_calibration_results.json. The verifier/generator is
../07_reduce_plot_calibration.py; exact plotted arrays and PDF SHA256 are in
../fast_calibration_plot_values.json. Calibration excludes mechanics smokes,
failed infrastructure starts and the unapplied terminal gradient from prefixes.

All loss/cosine/q curves are pointwise with no smoothing. Boundary histories
have129 states (0..128); update streams have128 accepted updates. A prefix has
A=sum(norm(x_i)), B=norm(sum(x_i)), R=B/A. Stored first moments use their actual
FP32 values after consuming each gradient; adaptive directions are reconstructed
in FP64 from bias-corrected Adam states. Actual updates use measured parameter
differences. Every unique trainable parameter is counted once.

The loss label uses the last64 boundary losses (63 adjacent changes), rise
tolerance1e-5 relative to previous loss. Smooth requires tail mean<=95% of
initial and rises<=5%. Oscillatory requires rises>=25%, median absolute relative
change>=0.001 and tail mean<=initial. Report continuous values alongside labels.
Tail stream medians use the last64 accepted updates; the terminal raw cosine
is independently computed and plotted at state128.

## Figure1: fast_calibration.pdf

Caption: Full training loss, adjacent raw-gradient cosine, normalized projected
gradient change q, and raw-gradient/stored-first-moment norms during128 fixed-LR
Adam updates from the same frozen initialization per case. Columns are CIFAR10
MLP (first5k images), CIFAR10 ResNet32 (all50k, fixed ghost-BN groups1000) and
WikiText2 Transformer (all2,088,600 prepared causal targets per update). Colors
identify three constant LRs in each case. Solid/dashed norm lines are raw/moment.
The q=1 reference means zero consecutive-gradient inner product; it is not an
Adam stability boundary. Every boundary value is shown, including startup spikes.
The terminal gradient at128 is measured but unapplied.

## Figure2: fast_calibration_streams.pdf

Caption: Adjacent cosines and complete-prefix B/A for raw gradients, stored
first moments, LR-free adaptive directions and measured parameter updates.
Colors denote the four streams; solid/dashed/dotted lines denote the lowest,
middle and highest LR within each case. Prefixes start at initialization and
cover128 accepted updates. Stored moments and adaptive vectors have distinct
time weighting and geometry. Constant LR makes ideal adaptive-step B/A equal
to adaptive-direction B/A; close agreement with actual updates is expected and
is not independent scientific corroboration. Overlapping adaptive/actual curves
are retained to show the measured rounding discrepancy.

## Coverage, uncertainty and interpretation

One fixed seed0 initialization per case, fresh zero Adam moments, beta=(0.9,0.999),
epsilon1e-8, no clipping/decay/warmup/dropout/augmentation/schedule. No Hessian,
HVP or spectral measurement. Training-only pilot; no validation/test or
uncertainty bands. Full batch removes changing-batch sampling, not dependence
on initialization. Initial tensors/data/reductions match Analysis024; floating
point execution need not be bitwise identical across machines.

MLP used one owned RTX5070Ti laptop across LRs. Every ResNet/Transformer LR used
RTX5090 class with the same pinned runtime and partition. Host CPU differences
still affect wall time. Source tensor hashes, starting gradients and all
retrieved files are checked. Final CPU vector checks independently reconstruct
moment/adaptive streams, q, complete-prefix B and parameter telescoping; numerical
reduction tolerance accounts for CPU/GPU FP64 summation order.

Raw-gradient reversal can coexist with decreasing loss and coherent adaptive
motion. Neither cosine sign nor q identifies sharpness, Hessian definiteness,
stability, generalization, or optimizer benefit. A128-update label is provisional;
long Adam second-moment startup and later regime changes remain unresolved.
The Analysis034 continuation calibration selected the grid; its trained-state
regimes were not assumed to transfer to initialization. See ../REPORT.md for
all nine results and missing brackets. No main trajectory was launched.

## Verification and visual review

Eleven focused tests pass. Manifest inventories, initial/final tensor hashes,
optimizer clocks, all prefix lengths/energies, final adaptive reconstruction,
terminal-gradient q and direct displacement identities are verified by
07_reduce_plot_calibration.py. Plotted arrays are checked against Matplotlib
artists before saving PDFs. Both one-page PDFs are rendered and visually reviewed
at closeout; scratch PNGs are excluded from Git. Exact figure hashes are recorded
in fast_calibration_plot_values.json.
