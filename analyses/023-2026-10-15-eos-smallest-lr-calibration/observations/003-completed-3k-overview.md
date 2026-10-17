# Completed three-case 3k GD overview

Initially requested October16 for the completed MLP/ResNet cases; updated
October17 to include the completed Transformer at the user's request.
The publication figure is
[eos_gd_overview_completed_3k.pdf](../figures/eos_gd_overview_completed_3k.pdf).
The existing Analysis020 figure and all manuscript files are unchanged.

## Sources, reduction and coverage

Run059: CIFAR-10 first5k / tanh200-200 MLP, LR0.004, seed0, first3,000
updates of the immutable20k trajectory. Run120: CIFAR-10 all50k / ResNet-32,
LR0.1, seed0, completed3,000-update run. Run121: WikiText-2 / two-layer
Transformer, LR0.05, seed0, completed3,000-update run, with2,088,600 supervised
training target tokens per update. All use plain full-batch GD, constant
LR, frozen initialization/data and the documented FP32/FP64 precision. The
ResNet uses50 fixed1k ghost-normalization groups; its objective is unchanged.
The three cases differ in architecture, loss and training data.

Run120's32 files /106,359,542bytes and Run121's32 files /3,017,412,264bytes
were retrieved from the owning Spark root
with remote/local archive and per-file SHA256/byte checks. The verified
temporary archives were removed; raw evidence and recovery states remain on
both machines. Run059's15-file legacy inventory was checked again:79 verified
files /3,231,509,850bytes across the three evidence inventories. Source,
config and inventory hashes appear in `completed_3k_results.json`.

Generator: `10_plot_completed_3k.py`. Uses the qualified Analysis020 loader and
Analysis021 retrieval verifier. The figure has three columns and four rows,
including raw-gradient norms added at the user's request. No temporal smoothing
or seed aggregation is applied. The first three rows use linear axes; norm
panels use logarithmic Y axes. Y limits are independent. Loss covers
states0..3000, including the actual final-state value;
the MLP endpoint comes from its next recorded preupdate row. Spectra cover
states0,100,...,3000. The projected-change row covers the2,999 defined adjacent
pairs per case at ending states1..2999; no extra gradient is inferred at3000.
The norm row covers all3,000 gradients at states0..2999 per case. Every norm
is finite and positive; no zero is clipped, floored or replaced for logarithmic
display. In total:9,000 update rows,8,997 pairs,93 plotted largest eigenvalues,
9,000 raw norm values and27,093 exact plotted points. The first adjacent pair
is undefined. Hashes confirm that all eight previously plotted MLP/ResNet
curves are unchanged by the Transformer addition.

Every used q=1-rho*norm(g_t)/norm(g_previous) identity and its exact reversal
boundary are checked. Prefix A, energy, lag-one accumulation, B-squared
decomposition and B/A also pass. All spectral eigenpairs pass the declared
relative-residual threshold0.005; all initial/config hashes match. MLP's legacy
solver requested two eigenvalues; only the largest is plotted. The new ResNet
and Transformer solver requested one; this diagnostic difference does not
alter GD updates.

## Caption

Full-batch GD at the smallest selected learning rate, seed0, through3,000
updates. Columns: CIFAR-10/tanh MLP (5k images, eta0.004), CIFAR-10/ResNet-32
(50k images, eta0.1), and WikiText-2/Transformer (2.09M training target tokens,
eta0.05). Rows: unsmoothed full-objective training loss;
eta times the largest algebraic Hessian eigenvalue, measured every100 updates;
unsmoothed gradient-derived directional curvature
eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous); and unsmoothed raw-gradient
norms on logarithmic Y axes. MLP spectra use the full5k
objective; ResNet spectra use the first five fixed1k ghost groups, explicitly
labeled as a5k subset. Transformer spectra use a fixed2,500-sequence subset.
Dashed lines mark eta*lambda=2 (GD quadratic reference)
and q=1 (exact consecutive-gradient reversal boundary), not global stability
guarantees. Lines between spectral markers do not supply unsampled spectra.

## Descriptive observations and limits

| Quantity | MLP | ResNet | Transformer |
| --- | ---: | ---: | ---: |
| Loss at state3000 | 0.336508 | 0.489838 | 6.630097 |
| First negative adjacent gradient update | Not observed | 231 (ending state230) | Not observed |
| Mean cosine, final1k pairs | 0.999999957 | -0.968385 | 0.999995184 |
| Mean q, final1k pairs | 0.0000437457 | 1.970699 | 0.000196315 |
| Maximum sampled eta*lambda_max | 0.316827 | 3.461436 | 1.722854 |
| Last sampled eta*lambda_max | 0.316827 | 3.248849 | 1.722854 |
| Final full-prefix B/A | 0.794144 | 0.037375 | 0.722349 |
| Initial raw-gradient norm | 2.392584 | 0.163048 | 1.289067 |
| Raw-gradient norm at state2999 | 0.0636365 | 2.033882 | 0.0516224 |

The MLP and Transformer remain positively adjacent-aligned in this window.
Transformer's sampled eta*lambda_max increases to1.722854 while its late q
remains near zero. ResNet develops
negative adjacent alignment while its objective continues to decrease overall;
its late q is close to2. The q/cosine correspondence is algebraic, not an
independent confirmation. No LR effect is isolated across these different
cases; one seed provides no ensemble uncertainty. ResNet/Transformer subset
spectra are not exact full-objective sharpness. No new training, horizon change, cloud
execution or manuscript edit was performed for this figure.

Current Poppler render was inspected for all twelve panels, unclipped tick labels,
logarithmic norm axes, LR/seed legends, spectral-coverage labels and margins. Programmatic checks
confirm exact artist arrays and one-page PDF text/coverage, including all twelve
panel letters, three model headings and the Transformer subset label. The existing
Analysis020 reference PDF still matches its recorded SHA256.
