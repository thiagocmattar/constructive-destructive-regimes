# Completed MLP/ResNet 3k GD overview

Requested October16 while the Transformer calibration remains in progress.
The separate publication figure is
[eos_gd_overview_completed_3k.pdf](../figures/eos_gd_overview_completed_3k.pdf).
The existing Analysis020 figure and all manuscript files are unchanged.

## Sources, reduction and coverage

Run059: CIFAR-10 first5k / tanh200-200 MLP, LR0.004, seed0, first3,000
updates of the immutable20k trajectory. Run120: CIFAR-10 all50k / ResNet-32,
LR0.1, seed0, completed3,000-update run. Both use plain full-batch GD, constant
LR, frozen initialization/data and the documented FP32/FP64 precision. The
ResNet uses50 fixed1k ghost-normalization groups; its objective is unchanged.
The two cases differ in architecture, loss and training-set size.

Run120's32 files /106,359,542bytes were retrieved from the owning Spark root
with remote/local archive and per-file SHA256/byte checks. The verified
temporary archives were removed; raw evidence and recovery states remain on
both machines. Run059's15-file legacy inventory was checked again. Source,
config and inventory hashes appear in `completed_3k_results.json`.

Generator: `10_plot_completed_3k.py`. Uses the qualified Analysis020 loader and
Analysis021 retrieval verifier. The figure has two columns and four rows,
including raw-gradient norms added at the user's request. No temporal smoothing
or seed aggregation is applied. The first three rows use linear axes; norm
panels use logarithmic Y axes. Y limits are independent. Loss covers
states0..3000, including the actual final-state value;
the MLP endpoint comes from its next recorded preupdate row. Spectra cover
states0,100,...,3000. The projected-change row covers the2,999 defined adjacent
pairs per case at ending states1..2999; no extra gradient is inferred at3000.
The norm row covers all3,000 gradients at states0..2999 per case. Every norm
is finite and positive; no zero is clipped, floored or replaced for logarithmic
display. In total:6,000 update rows,5,998 pairs,62 plotted largest eigenvalues,
6,000 raw norm values and18,062 exact plotted points. The first adjacent pair
is undefined. Hashes confirm that the original six plotted curves are unchanged.

Every used q=1-rho*norm(g_t)/norm(g_previous) identity and its exact reversal
boundary are checked. Prefix A, energy, lag-one accumulation, B-squared
decomposition and B/A also pass. All spectral eigenpairs pass the declared
relative-residual threshold0.005; all initial/config hashes match. MLP's legacy
solver requested two eigenvalues; only the largest is plotted. The new ResNet
solver requested one; this diagnostic difference does not alter GD updates.

## Caption

Full-batch GD at the smallest selected learning rate, seed0, through3,000
updates. Columns: CIFAR-10/tanh MLP (5k images, eta0.004) and CIFAR-10/ResNet-32
(50k images, eta0.1). Rows: unsmoothed full-objective training loss;
eta times the largest algebraic Hessian eigenvalue, measured every100 updates;
unsmoothed gradient-derived directional curvature
eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous); and unsmoothed raw-gradient
norms on logarithmic Y axes. MLP spectra use the full5k
objective; ResNet spectra use the first five fixed1k ghost groups, explicitly
labeled as a5k subset. Dashed lines mark eta*lambda=2 (GD quadratic reference)
and q=1 (exact consecutive-gradient reversal boundary), not global stability
guarantees. Lines between spectral markers do not supply unsampled spectra.

## Descriptive observations and limits

| Quantity | MLP | ResNet |
| --- | ---: | ---: |
| Loss at state3000 | 0.336508 | 0.489838 |
| First negative adjacent gradient update | Not observed | 231 (ending state230) |
| Mean cosine, final1k pairs | 0.999999957 | -0.968385 |
| Mean q, final1k pairs | 0.0000437457 | 1.970699 |
| Maximum sampled eta*lambda_max | 0.316827 | 3.461436 |
| Last sampled eta*lambda_max | 0.316827 | 3.248849 |
| Final full-prefix B/A | 0.794144 | 0.037375 |
| Initial raw-gradient norm | 2.392584 | 0.163048 |
| Raw-gradient norm at state2999 | 0.0636365 | 2.033882 |

The MLP remains positively adjacent-aligned in this window. ResNet develops
negative adjacent alignment while its objective continues to decrease overall;
its late q is close to2. The q/cosine correspondence is algebraic, not an
independent confirmation. No LR effect is isolated across these different
cases; one seed provides no ensemble uncertainty. ResNet's subset spectrum is
not the exact full-objective sharpness. The Transformer is excluded, not shown
as a partial or completed condition. No new training, horizon change, cloud
execution or manuscript edit was performed for this figure.

Current Poppler render was inspected for all eight panels, unclipped tick labels,
logarithmic norm axes, LR/seed legends, spectral-coverage labels and margins. Programmatic checks
confirm exact artist arrays and one-page PDF text/coverage. The existing
Analysis020 reference PDF still matches its recorded SHA256.
