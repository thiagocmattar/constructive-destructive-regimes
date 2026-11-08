> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-paired-displacement.md](../../../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md).

# Observation001: local reversal and cumulative motion

## Question and sources

How does strong consecutive reversal coexist with nonzero displacement and
training-loss reduction? Separate cancellation within adjacent pairs from
turning between their residual displacements. Compare the normalized path
ratio with its zero-cross-term reference, rather than interpreting a small
ratio as negative cumulative interaction or a fraction of retained learning.

Sources are the nine Analysis024 full-GD conditions (Runs059/062/120--126),
all nine GD backward profiles in Analysis035/Run212, and Analysis037's moving
SGD comparison, including the unchanged Analysis027 MLP continuations.
`../results.json` records 43 source hashes. Scientific inputs and raw records
are unchanged; earlier source manifests retain model/data/config identities.
This is local retrospective reduction, with no training or cloud execution.

## Reduction and coverage

`../01_analyze.py` uses all 27,000 applied gradient observations and 26,991
defined consecutive pairs. Norms, dot products and prefix statistics cover
all trainable parameters; FP32 training and FP64 geometry follow the owning
runs. Loss covers the complete fixed training objective with its original
example/token weighting. Final-state loss is included; no validation subset
or sampling extrapolation is introduced.

The [analysis contract](../../../../../analyses/056-2026-11-05-gd-paired-displacement/README.md) defines C, D, R, R_perp and Q and their
zero-denominator behavior. Pair norms use raw observations and the numerically
stable norm identity. Primary partitions pair updates 1/2, 3/4, ..., 2999/3000.
The shifted partition pairs 2/3, ..., 2998/2999 and retains updates 1 and 3000
as singletons, preserving the same full-path displacement. This is a phase
sensitivity check, not a claim to recover a cropped-window vector from scalar
prefix norms. Prefix factors are evaluated at every even T.

Both offsets' complete two-update loss changes are retained in ignored array
artifacts. `../loss_windows.csv` summarizes the full prefix and the fixed final
1,000-update window, with strict-sign counts and relative 1e-6 sensitivity.
The shifted loss windows exclude the two unmatched boundary updates and have
one fewer pair. No event or sign filtering precedes reduction.

All 2,304 GD backward pairs retain a fixed endpoint per curve; CPU and GPU
products agree with the reported cosine. The three endpoint states were
chosen in the earlier Analysis035 design, before this reanalysis. The
2000-start profile remains the detailed main trace. Summary means average
odd/even lags 1--32 separately within each endpoint, without pooling endpoints.

SGD summaries give equal weight to three initialization blocks after averaging
eight stochastic repetitions (one for full GD) within each block. Each
continuation contributes 199 pairs. The independent check reaggregates all
546 retained continuations into 28 cells; shades in the unchanged SGD figure
are initialization-block ranges, not confidence intervals.

## Findings

At the highest rates:

| Model | C | D | R | R_perp | Q | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP, 0.04 | 0.11386 | 0.22544 | 0.02567 | 0.01993 | 1.65863 | 0.153869 |
| ResNet, 0.1 | 0.15579 | 0.23991 | 0.03737 | 0.02019 | 3.42657 | 0.489838 |
| Transformer, 0.2 | 0.08592 | 0.63002 | 0.05413 | 0.01838 | 8.67063 | 6.069747 |

Each largest-rate run has its model's lowest final training loss among the
tested rates. Immediate cancellation removes most path length, but appreciable
turning remains between pairs, especially for MLP and ResNet. The Transformer
has more coherent residual accumulation in this comparison. Both factors
change by less than 0.004 for these three runs when shifting the partition.
All nine conditions, not only these three, have Q>1: cumulative interaction
across all pairs is positive. Seven have negative cumulative adjacent
interaction. `../endpoints.csv` includes every rate and both offsets.

Across states 2000--3000 at the highest rates, primary pairs decrease loss
246/500, 248/500 and 386/500 times, respectively. Total changes are
-0.064411, -0.304780 and -0.186658. Alternate interior pairs decrease loss
254/499, 244/499 and 374/499 times, with negative totals over 2001--2999.
The 1e-6 sensitivity leaves all primary late-window counts unchanged and
removes two near-zero alternate Transformer changes. Pairing is a geometric
operation and does not make individual two-update loss changes monotone.

All nine backward profiles have negative odd-lag and positive even-lag means
over lags 1--32. Only the Transformer retains the corresponding sign at every
one of these lags at all three endpoints. The MLP and ResNet envelopes contract
and have sign exceptions; full-lag envelopes can strengthen again or change
phase. Alternation is temporal organization, not proof of nonzero displacement.

The three SGD trained-state full-GD means (-0.991/-0.967/-0.993) weaken to
-0.384/-0.612/-0.258 at the smallest tested batches, in every block. Initial
full-GD means are positive for MLP/ResNet and negative for Transformer. Small
MLP batches change sign in every block and vary nonmonotonically. These moving
paths combine sampling variability with changes in visited states. They do
not isolate attenuation of a fixed negative full-gradient cosine.

## Figure and caption contracts

Generated by `../03_plot.py`; copies in `manuscript/draft/figures/analysis056/`.
`../figure_manifest.json` binds all 99 curves / 191,826 plotted observations,
their array hashes, source reduction and four vector PDFs. Scratch PNGs are
ignored. The active captions are in `sections/04-controlled-figures.tex` and
`sections/07-gd-diagnostics.tex`.

- `gd_loss_curvature.pdf`: unsmoothed full-objective loss at states 0--3000;
  complete 50-pair trailing secant means; all 279 retained spectral points.
  One and two are distinct reference thresholds. Spectra are placed at s+2
  to align their starting state with the secant's gradient index. Full MLP,
  fixed 5k/50k ResNet and 2500/59680 Transformer spectral coverage remains
  explicit. Direction, step averaging and objective coverage all differ.
- `gd_reversal_paired_displacement.pdf`: four rows of cosine, gradient norms,
  R/R_perp, and C/D. Only the first two rows use 50-observation trailing means.
  Norms and path ratios have log axes; factors have linear axes. All nine
  conditions remain visible. Solid/dotted distinguish R/reference and
  solid/dashed distinguish C/D, with consistent learning-rate colors.
- `gd_lag_endpoints.pdf`: original 2256-endpoint exact traces at lags 1--256
  and 1--32, plus separate odd/even means at 756/2256/3256. Connecting lines
  are visual guides, not interpolation evidence. No uncertainty band.
- `gd_two_step_loss_changes.pdf`: every primary/shifted two-update loss change
  in the last 1,000 updates of the highest-rate runs, 500/499 per case.
  Separate vertical scales and zero lines; no smoothing or exclusion.

The appendix table is generated from all nine endpoints and both partitions.
Previously published curves agree with 133,668 retained source values to
maximum absolute difference 8.89e-16; additional curves are new reductions,
not new observations. Prior figures are preserved byte for byte.

## Verification, uncertainty and interpretation limits

`../02_verify.py` passes 65 direct-vector examples, including aligned,
orthogonal, exactly cancelling and all-zero cases. All 26,991 pair norms
agree with an independent 45-digit decimal reconstruction (maximum absolute
error 1.06e-15). Prefix identities, both partitions, loss telescoping,
index placement, all raw-source hashes and block-first aggregation pass.
Recorded actual displacement norms and eta*B agree to relative error
9.52e-8 or better, which checks norms rather than exact vector replay.

One initialization per GD model supplies no initialization uncertainty.
The lag profiles are separate recomputations with known later FP32 drift.
SGD also has runtime/initialization confounding in the two smallest seed-2
ResNet batches, documented by Analysis037. No pair or timepoint is treated
as independent seed replication. These results do not establish stability,
generalization, causal benefit from reversal, learned-representation
overwriting or a validated controller. A fixed-two-endpoint sampling control
and same-objective directional HVP check remain optional new measurements;
this manuscript makes neither corresponding validation claim.
