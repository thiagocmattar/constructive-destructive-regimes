# Observation003: completed production runs, interim preview

## Scope and sources

The user requested a preview using only completed setups. This figure uses the
four complete, locally retrieved and scientifically verified Run215 arms in
`runs/215-2026-11-02-controlled-full-batch-adam/results.json`: MLP at learning
rates0.0001/0.0003/0.001 and ResNet-32 at0.003. Each contains1,024 updates.
The other two ResNet arms and three Transformer arms are omitted. Calibration
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
increasing-LR color assignment, so the single completed ResNet rate is green.
The Transformer column is explicitly pending. Coverage is3/3 MLP,1/3 ResNet,
0/3 Transformer; blank panels do not indicate a measured zero.

## Reduction and timing

The approved Observation002 definitions apply unchanged. The first seven rows
use complete16-observation trailing arithmetic means; no padding, extrapolation
or incomplete starting windows. The raw-gradient expression is computed pointwise
before smoothing. Log scaling follows averaging. R_t is the unsmoothed norm of
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
The MLP arms share the owned5070Ti laptop; the completed ResNet arm uses anRTX4090.
Other hardware is recorded in Run215 and does not enter this preview. No Hessian,
LR intervention, training change, new cloud allocation or scientific claim is added.
This incomplete cohort does not support a comparison across all three cases or
all ResNet/Transformer learning rates. Smoothing suppresses short fluctuations
and delays/broadens changes; unsmoothed histories remain preserved.

All32 curves /32,336 plotted values pass source and artist checks. Independent
scalar math.fsum means agree within8.89e-16 absolute error. Every R_t prefix equals
the stored source value. The one-page,24-panel PDF passes text-boundary and footer
separation checks. Its final render was visually reviewed: no clipped labels,
overlapping text or obscured curves; empty Transformer panels are marked pending.
The PNG render is ignored scratch, while the publication artifact is PDF.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/10_plot_completed_preview.py

Output: `figures/adam_diagnostics_completed_preview.pdf`. Re-running selects the
complete verified cohort available at that later time; this manifest preserves
the present four-condition snapshot and its exact plotted arrays.
