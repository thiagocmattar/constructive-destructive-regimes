# Observation003: completed production runs, interim preview

## Scope and sources

The user requested a preview using only completed setups. This figure uses the
six complete, locally retrieved and scientifically verified Run215 arms in
`runs/215-2026-11-02-controlled-full-batch-adam/results.json`: MLP at learning
rates0.0001/0.0003/0.001, ResNet-32 at0.0003/0.003 and Transformer at0.0003.
Each contains1,024 updates. The remaining ResNet rate and two Transformer rates
are omitted. Calibration
prefixes and active partial trajectories are not joined to these histories.

`10_plot_completed_preview.py` verifies the cohort manifest hashes and source
history bytes/SHA256 before plotting. `completed_preview_manifest.json` records
all source identities, original/plotted values, coverage, generator and PDF hashes.
`completed_preview_verification.json` records independent numerical and PDF checks.
The original approved calibration figure remains unchanged.

## Caption

Completed-run preview of controlled full-batch Adam training over1,024 updates.
Columns are CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer. The eight
rows show training loss (log scale), raw-gradient eta*kappa_hat, consecutive
cosine rho for raw gradients, momentum and adaptive Adam directions, raw-gradient
norm and stored first-moment norm (separate log-scale rows), and cumulative R_t
for Adam's measured parameter updates. Blue/orange/green retain the full grid's
increasing-LR color assignment. Coverage is3/3 MLP,2/3 ResNet,1/3 Transformer.
Training loss and Adam R_t are unsmoothed; the other six rows use16-step trailing
means. Missing rates have no trace.

## Reduction and timing

The user revised the display to keep both training loss and Adam R_t unsmoothed.
All1,025 measured training losses are plotted at states0..1024, without averaging
or omission. The six diagnostic rows retain complete16-observation trailing
arithmetic means; no padding, extrapolation or incomplete starting windows.
Observation002 metric definitions apply unchanged. The raw-gradient expression
is computed pointwise before smoothing. Log scaling follows averaging in the
smoothed norm rows; raw loss receives only log scaling. R_t is the unsmoothed norm of
net measured Adam displacement divided by cumulative measured update length.
No raw-gradient or momentum R_t is displayed.

Loss states are0..1024. Vector rows label applied updates1..1024, consuming raw
gradients from states0..1023. The separately sampled terminal gradient is excluded
from vector rows. Consecutive cosines and projected change are undefined at the
first update. The horizontal coordinate is Training step. The two norm rows use
the same vertical range within a column. The dashed reference at2 is retained
from the GD figure; it is not an identified Adam stability threshold.

## Coverage, uncertainty and verification

One frozen initialization per case, no seed averaging or uncertainty bands.
The MLP arms share the owned5070Ti laptop; ResNet LR0.0003 uses anRTXPRO4000,
ResNet LR0.003 and Transformer LR0.0003 use RTX4090 workers. Each cloud worker
passed numerical start qualification, but different ResNet GPU models limit
attributing differences exclusively to LR. No Hessian,
LR intervention, training change, new cloud allocation or scientific claim is added.
This incomplete cohort does not cover all ResNet/Transformer learning rates.
Smoothing suppresses short fluctuations and delays/broadens changes in its six
rows; it is not applied to loss or R_t. Original histories remain preserved.

All48 curves /48,594 plotted values pass source and artist checks. Independent
scalar math.fsum means agree within8.89e-16 absolute error. Every R_t prefix equals
the stored source value; every loss value and its state equal the source exactly.
The one-page,24-panel PDF passes text-boundary and footer
separation checks. Its final render was visually reviewed: no clipped labels,
overlapping text or obscured curves. All three columns now contain complete runs.
The PNG render is ignored scratch, while the publication artifact is PDF.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/10_plot_completed_preview.py

Output: `figures/adam_diagnostics_completed_preview.pdf`. Re-running selects the
complete verified cohort available at that later time; this manifest preserves
the present six-condition snapshot and its exact plotted arrays.
