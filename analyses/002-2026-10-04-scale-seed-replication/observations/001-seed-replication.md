# Observation 001: Scale geometry across three seeds

Status: completed full passes; raw evidence retrieved and independently verified.

The user requested two additional seeds at each size to examine a possible single-run effect.
Seeds 1235 and 1236 vary both initialization and training-data permutation, paired across sizes.
Seed1234 comes from the existing paired runs; every size now has three trajectories.

## Sources and reduction

[results.json](../results.json) retains exact source/config/environment/acceptance hashes,
[figure_data.csv](../figure_data.csv) retains all 6,408 per-update reductions, and
[01_compare.py](../01_compare.py) generates all statistics and publication PDFs.
[02_record.py](../02_record.py) writes this observation. The pilot reduction functions
are reused with their script hash recorded; no old raw evidence is modified.

- 14M, seed 1234: [Run002](../../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/README.md), attempt-02; 14,067,712 parameters; NVIDIA GeForce RTX 4090.
- 31M, seed 1234: [Run003](../../../runs/003-2026-10-03-pythia31m-adamw-scale/README.md), attempt-01; 30,494,720 parameters; NVIDIA GeForce RTX 4090.
- 70M, seed 1234: [Run004](../../../runs/004-2026-10-03-pythia70m-adamw-scale/README.md), attempt-01; 70,426,624 parameters; NVIDIA GeForce RTX 4090.
- 14M, seed 1235: [Run005](../../../runs/005-2026-10-04-pythia14m-seed1235/README.md), attempt-01; 14,067,712 parameters; NVIDIA GeForce RTX 5090.
- 31M, seed 1235: [Run006](../../../runs/006-2026-10-04-pythia31m-seed1235/README.md), attempt-01; 30,494,720 parameters; NVIDIA GeForce RTX 5090.
- 70M, seed 1235: [Run007](../../../runs/007-2026-10-04-pythia70m-seed1235/README.md), attempt-01; 70,426,624 parameters; NVIDIA GeForce RTX 5090.
- 14M, seed 1236: [Run008](../../../runs/008-2026-10-04-pythia14m-seed1236/README.md), attempt-01; 14,067,712 parameters; NVIDIA GeForce RTX 5090.
- 31M, seed 1236: [Run009](../../../runs/009-2026-10-04-pythia31m-seed1236/README.md), attempt-01; 30,494,720 parameters; NVIDIA GeForce RTX 5090.
- 70M, seed 1236: [Run010](../../../runs/010-2026-10-04-pythia70m-seed1236/README.md), attempt-01; 70,426,624 parameters; NVIDIA GeForce RTX 5090.

All trajectories match token/cache bytes, 712 updates, effective batch1024/microbatch4,
the AdamW/LR/clipping recipe, FP16 scaling, parameter scope, loss weighting, and validation.
Exact permutation hashes match across sizes within each seed and differ across seed blocks.
Architectural width/head configuration and initialization scale vary with model size.
No pretrained weights are loaded. Each pass consumes all 1,491,711,416 input tokens once.
Initial/final validation includes all 500 documents and the packed short tail, token-weighted.

For x=g (unscaled raw gradient before clipping) or d (adaptive Adam direction from actual
updated FP32 moments of the clipped gradient, evaluated in FP64), A_t=sum_{i<=t}||x_i||,
B_t=||sum_{i<=t}x_i|| and R=B/A. d excludes learning-rate scaling and weight decay.
Cosines use adjacent accumulated optimizer boundaries. First cosine is undefined;
medians use 711 defined pairs. A/B/R are terminal at update712. Norm medians use all712
updates, with equal update weighting. Steps are not independent seed replicates.

## Primary replication comparison

Before the new full passes, the pilot expectation was fixed as decreasing median adjacent
cosine and terminal B/A with size, separately for raw and adaptive streams.

| Stream | Metric | Strict pilot ordering in new seeds | All three seeds |
| --- | --- | ---: | ---: |
| raw | median_adjacent_cosine | 2/2 | 3/3 |
| raw | R | 2/2 | 3/3 |
| adaptive | median_adjacent_cosine | 2/2 | 3/3 |
| adaptive | R | 2/2 | 3/3 |

The full-pass median-cosine and terminal-ratio ordering survives both new seed blocks
for both vector streams. This supports reproducibility of the pilot ordering under this
matched recipe. Adaptive cosine varies much less across seeds than raw cosine.

The counts are descriptive replication outcomes, not a p-value or a validated scaling law.
Any reversal or overlap is retained below rather than selecting successful seeds.

| Seed | Stream | 14M cosine | 31M cosine | 70M cosine | 14M B/A | 31M B/A | 70M B/A |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | raw | 0.007220 | -0.086252 | -0.160566 | 0.337132 | 0.330751 | 0.309380 |
| 1234 | adaptive | 0.966250 | 0.952865 | 0.931795 | 0.704780 | 0.665558 | 0.609436 |
| 1235 | raw | -0.003979 | -0.033065 | -0.111253 | 0.344891 | 0.324771 | 0.308670 |
| 1235 | adaptive | 0.966306 | 0.952479 | 0.932576 | 0.705245 | 0.659127 | 0.609304 |
| 1236 | raw | 0.013478 | -0.121242 | -0.129404 | 0.332470 | 0.315224 | 0.296680 |
| 1236 | adaptive | 0.966163 | 0.953217 | 0.931264 | 0.701689 | 0.663647 | 0.605221 |

## Secondary ordering and phase sensitivity

The pilot absolute-geometry directions are checked separately from the primary cosine/ratio
expectations. These dimensional quantities need not share the normalized ordering.

| Stream | Metric | Pilot direction | New seed blocks matching | All seed blocks matching |
| --- | --- | --- | ---: | ---: |
| raw | A | decrease | 1/2 | 2/3 |
| raw | B | decrease | 2/2 | 3/3 |
| raw | median_norm | No monotone pilot expectation | N/A | N/A |
| adaptive | A | increase | 2/2 | 3/3 |
| adaptive | B | increase | 2/2 | 3/3 |
| adaptive | median_norm | increase | 2/2 | 3/3 |

Raw A is nonmonotone in seed1235: 14M/31M/70M values are 318.675272/301.702222/302.241590. This small reversal is retained.

Full-pass ordering does not imply ordering at every phase. Quarter cosine medians
use pairs whose later update lies in the specified quarter (including the cross-boundary pair).
The following counts examine strict 14M > 31M > 70M ordering within each quarter.

| Later updates | Raw cosine ordering | Adaptive cosine ordering |
| --- | ---: | ---: |
| 1-178 | 2/3 | 3/3 |
| 179-356 | 0/3 | 3/3 |
| 357-534 | 1/3 | 3/3 |
| 535-712 | 3/3 | 3/3 |

## Mean, seed variability, and absolute geometry

Mean ± sample SD across three per-trajectory summaries; SD is seed spread, not a confidence interval.
The full JSON also provides each seed value, min/max, paired size differences and quarter summaries.

| Model | Stream | Median cosine | Terminal B/A | Terminal A | Terminal B | Median norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 14M | raw | 0.00557295 ± 0.00884 | 0.338164 ± 0.00627 | 320.156 ± 1.91 | 108.258 ± 1.46 | 0.410591 ± 0.00199 |
| 14M | adaptive | 0.96624 ± 7.21e-05 | 0.703905 ± 0.00193 | 947271 ± 1.35e+03 | 666789 ± 2.57e+03 | 1247.5 ± 2.17 |
| 31M | raw | -0.0801861 ± 0.0444 | 0.323582 ± 0.00783 | 307.322 ± 6.4 | 99.4192 ± 1.64 | 0.387123 ± 0.0163 |
| 31M | adaptive | 0.952853 ± 0.000369 | 0.662778 ± 0.0033 | 1.20541e+06 ± 4.05e+03 | 798928 ± 6.6e+03 | 1625.39 ± 2.62 |
| 70M | raw | -0.133741 ± 0.0249 | 0.30491 ± 0.00714 | 304.177 ± 1.76 | 92.7409 ± 1.84 | 0.383746 ± 0.00908 |
| 70M | adaptive | 0.931878 ± 0.00066 | 0.607987 ± 0.0024 | 1.56008e+06 ± 5.05e+03 | 948515 ± 6.78e+03 | 2127.27 ± 7.29 |

| Run | Seed | Model | Vector | A | B | B/A | Median norm | Negative pairs |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 002 | 1234 | 14M | raw | 319.486529 | 107.70908 | 0.337132 | 0.409031398 | 350/711 |
| 002 | 1234 | 14M | adaptive | 946630.44 | 667166.451 | 0.704780 | 1250.00572 | 0/711 |
| 003 | 1234 | 31M | raw | 305.977 | 101.20235 | 0.330751 | 0.386882817 | 405/711 |
| 003 | 1234 | 31M | adaptive | 1207840.6 | 803888.444 | 0.665558 | 1627.64857 | 0/711 |
| 004 | 1234 | 70M | raw | 304.620311 | 94.243503 | 0.309380 | 0.394048379 | 481/711 |
| 004 | 1234 | 70M | adaptive | 1563927.22 | 953113.16 | 0.609436 | 2132.92633 | 0/711 |
| 005 | 1235 | 14M | raw | 318.675272 | 109.908108 | 0.344891 | 0.412825404 | 357/711 |
| 005 | 1235 | 14M | adaptive | 948818.038 | 669148.767 | 0.705245 | 1246.15022 | 0/711 |
| 006 | 1235 | 31M | raw | 301.702222 | 97.9841841 | 0.324771 | 0.370976794 | 379/711 |
| 006 | 1235 | 31M | adaptive | 1200740.94 | 791440.963 | 0.659127 | 1622.51964 | 0/711 |
| 007 | 1235 | 70M | raw | 302.24159 | 93.2928425 | 0.308670 | 0.380257393 | 459/711 |
| 007 | 1235 | 70M | adaptive | 1561950.46 | 951702.718 | 0.609304 | 2129.85409 | 0/711 |
| 008 | 1236 | 14M | raw | 322.305748 | 107.157112 | 0.332470 | 0.409915921 | 348/711 |
| 008 | 1236 | 14M | adaptive | 946363.036 | 664052.59 | 0.701689 | 1246.35636 | 0/711 |
| 009 | 1236 | 31M | raw | 314.287822 | 99.0711876 | 0.315224 | 0.403508475 | 459/711 |
| 009 | 1236 | 31M | adaptive | 1207653.75 | 801456.021 | 0.663647 | 1625.99216 | 0/711 |
| 010 | 1236 | 70M | raw | 305.670511 | 90.6862819 | 0.296680 | 0.376931762 | 485/711 |
| 010 | 1236 | 70M | adaptive | 1554359.95 | 940730.527 | 0.605221 | 2119.04076 | 0/711 |

| Model | Seed | Median paired ||d||/||g|| | Initial validation loss | Final validation loss | Overflow retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14M | 1234 | 3175.945728 | 11.032736 | 5.182392 | 0 |
| 31M | 1234 | 4209.503357 | 11.002269 | 4.611768 | 0 |
| 70M | 1234 | 5382.225966 | 11.063131 | 4.063201 | 0 |
| 14M | 1235 | 3131.877064 | 11.027963 | 5.182016 | 0 |
| 31M | 1235 | 4410.822571 | 11.003804 | 4.617403 | 0 |
| 70M | 1235 | 5597.969275 | 10.983395 | 4.058427 | 0 |
| 14M | 1236 | 3149.895725 | 11.013281 | 5.164867 | 0 |
| 31M | 1236 | 4080.573208 | 11.020435 | 4.522797 | 0 |
| 70M | 1236 | 5577.823063 | 11.004331 | 4.029232 | 0 |

## Figures and coverage

- [Geometry evolution](../figures/geometry_across_seeds.pdf): adjacent cosine, cumulative B/A and individual norm. Bold lines are means across seeds; faint lines are individual trajectories and shading is the seed range. Cosine curves first use a centered31-step median within each seed; adaptive cosine has a zoomed vertical axis. Norm axes are logarithmic.
- [Size summary](../figures/scale_summary_across_seeds.pdf): full-pass per-seed points connected by seed, plus means and sample SD, for cosine, B/A, A, B, norms and negative-pair fractions. Connecting lines are guides, not fitted laws. Absolute norms/A/B use log axes.
- [Cumulative A/B](../figures/cumulative_across_seeds.pdf): raw and adaptive cumulative sums of norms and norms of cumulative sums, with means and seed ranges.
- [Norm ratios](../figures/norm_ratio_across_seeds.pdf): per-step adaptive/raw norm ratios, with seed means/ranges and a logarithmic axis.

All evolution displays use updates1–711; the last partial batch (311 sequences) is omitted
only from display, as previously requested. All raw updates, CSV rows and full terminal
statistics retain update712. results.json also includes update711 statistics for sensitivity.
Envelope widths are descriptive min/max across only three trajectories, not statistical intervals.

## Interpretation limits

Three seeds are a small replication check. Pairing makes size differences interpretable
within each seed, but does not isolate initialization from data-order variability.
Seed1234 used RTX4090; the two new seed blocks use RTX5090. Software/precision/optimizer
contracts match, but hardware/kernel floating-point differences can change trajectories.
Within each new seed block all three sizes use the same GPU type.
Width, heads and initialization scale co-vary with parameter count; absolute A/B and norms
depend on dimensionality and parameterization. No universal power law, out-of-sample
prediction, stability, generalization, or causal learning effect is established.
Positive adjacent direction cosine alone does not imply collinearity over the full pass
or A≈B; long-horizon turning can leave a gap. Loss is measured independently.
These are vector-geometry diagnostics. The manuscript's constant-step plain-SGD parameter
norm identity is not applied directly to this AdamW trajectory with a varying LR and decay.
