# Completed Pythia preconditioning experiment

All27 runs completed: nine350-update baselines and eighteen50-update branches,
covering14M/31M/70M and seed blocks1234/1235/1236. All4,050 accepted updates,
scalar/probe histories and45 retained recovery states were downloaded. Local
size/SHA-256 and scientific checks passed for every run; retained artifacts total
103,516,215,158 bytes. The reduction reproduces exactly from the verified scalars.

Both task-owned pods and the200GB network volume were deleted on October17 at
19:05 BRT. Fresh Runpod reads confirm all three resources are absent. Posted
task charges total **USD11.416828**, against the approved USD25 cap; see
[provider records](billing.json). These are posted charges, subject to provider
reconciliation. No resources from other experiments were changed.

## Primary result

Removing Adam coordinate preconditioning increased fixed-probe loss in all18
branches, while following the unmodified parent's calendar-matched adaptive-step
norm schedule. The effect was larger for70M than14M in all three seed blocks at
both forks;31M was intermediate in each paired block.

The outcome is tau = ln F_ablated(end) - ln F_native(end), after50 updates from
the same checkpoint. Positive tau means worse loss after removing preconditioning.

| Size | Mean tau, fork175 | Mean tau, fork225 |
| --- | ---: | ---: |
| 14M | 0.208094 | 0.237750 |
| 31M | 0.350102 | 0.456766 |
| 70M | 0.522908 | 0.725101 |

The corresponding geometric mean ablated/native loss ratios at fork225 are
approximately1.27,1.58 and2.06. Every seed effect, paired size interaction,
native log-loss change and source hash is in [results.json](results.json).

The native fixed-probe log-loss changes over updates226–275 averaged
-0.02469/-0.02794/-0.05178 for14M/31M/70M. The larger model thus retained faster
relative improvement on this common probe in that window. Sparse one-step
decreases vary considerably across batches/seeds and do not uniformly rank sizes.

## What the direction probes explain

Same-state interventions expose a large finite-step nonlinear penalty when
preconditioning is removed. At update200, the70M seed means were:

| Direction | Linear prediction P/F | Nonlinear remainder R/F | Actual decrease D/F |
| --- | ---: | ---: | ---: |
| Native AdamW | 0.011986 | 0.011160 | 0.000826 |
| No preconditioning, matched adaptive norm | 0.050101 | 1.412979 | -1.362878 |

Here D=P-R, and negative D denotes a loss increase. The unpreconditioned
momentum direction has a larger favorable linear prediction in this comparison,
but its nonlinear penalty overwhelms that prediction. Removing first-moment
averaging also harms these one-step outcomes. Native/reconstructed Adam numerical
controls passed at every measurement point, so these large differences are far
larger than the checked reconstruction errors.

This supports an optimizer-mediated contribution: at the native adaptive-step
scale, Adam's coordinate scaling permits useful finite steps where the substituted
directions incur substantial nonlinear loss increases. The sustained ablations
show a larger dependence on that scaling in the larger model under this recipe.

This experiment does not identify a unique event at step200, establish a causal
capacity effect, or test an optimally retuned scalar learning rate for momentum.
The forks bracket overlapping portions of the originally selected200–250 window.
R includes higher-order terms and numerical error; it is not a measured Hessian
curvature or an edge-of-stability statistic. Global adaptive norms are matched
within each model, before FP32 rounding; decay and subsequent trajectory changes
remain part of the actual full parameter displacement.

## Figures and evidence

- [Native fixed-probe loss and D/P/R](figures/native_descent_accounting.pdf).
- [Paired50-update preconditioning effects](figures/preconditioning_effects.pdf).
- [Same-state direction and step-length substitutions](figures/direction_substitutions.pdf).

All three PDFs were rendered and visually inspected. The
[observation](observations/001-preconditioning-effects.md) records sources,
coverage, reductions, captions and caveats. The generating script is
[08_analyze.py](08_analyze.py); its complete-cohort reduction checks scalar
hashes against accepted inventories before analysis. No display smoothing is used.

Uncertainty is across three seed blocks; updates and the two forks are not
independent replicates. Initialization and data ordering co-vary within a block.
The fixed128-block probe is disjoint from the union of all training prefixes but
belongs to the same training corpus; it is not an independent test population.
Complete original validation coverage is retained in each run's summary, but the
primary causal endpoint uses calendar-matched fixed-probe losses. The motivating
interval was selected post hoc. No manuscript claims were edited.
