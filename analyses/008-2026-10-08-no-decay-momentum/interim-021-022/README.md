# Runs021/022: completed warmup-only learning-rate comparison

At the same full-pass budget, LR 0.001 reaches validation loss 4.763734, versus 6.232207 at LR 0.0001: a 1.468473-nat difference favoring Run021.

Raw adjacent gradients are much more often negative at the larger LR, while stored momentum and adaptive directions have positive adjacent cosines in both runs. The ordering of cumulative cancellation differs by stream: the smaller LR has higher raw and momentum B/A, but lower adaptive B/A. Thus greater geometric alignment does not consistently rank these two runs by validation performance.

[Six-panel figure](figures/six_panel_comparison.pdf) | [Coarse lag profiles](figures/checkpoint_lag_profiles.pdf) | [Checkpoint cosine matrices](figures/checkpoint_cosine_matrices.pdf)

The simplified versions are [metrics compared across learning rates](figures/metrics_by_learning_rate.pdf)
and [geometry within each learning rate](figures/geometry_within_learning_rate.pdf).
The first is 3x2, separating raw/momentum/Adam cosine and raw/momentum individual
norms into their own panels, with both LRs overlaid in each. The second is 2x2:
one column per LR, raw/momentum A/B above and raw/momentum/Adam B/A below.
Colors encode LR in the first figure and vector stream in the second. Norms use
linear axes; cosine ranges and geometry scales are matched across their panels.
These versions preserve the numerical findings and coverage below. See
[Observation002](observations/002-simplified-layouts.md) for the captions and
[data](simplified_figure_data.json) for sources and reductions.

The generator `../08_plot_simplified.py` accepts a third completed, verified run.
With `--runs 021 022 023 --output complete-three-lr`, it adds the third LR to
every metric panel and adds a third geometry column, producing a 2x3 geometry
figure. The current versions include only completed Runs021/022.

## Loss and coverage

The focused [2x2 loss/alignment/norm figure](figures/loss_alignment_norms.pdf)
overlays both LRs for loss and raw adjacent cosine, then compares raw and
momentum norms within each LR. Shading starts at the first negative unsmoothed
raw pair (updates 54 and 385), including later positive pairs. It marks time
after an event, not a negative regime or a defined onset of norm separation.
Momentum/raw norm ratios are already 0.627 and 0.610 one update earlier, while
raw adjacent cosine is still positive. Raw-norm fluctuations rise near the event
while momentum stays smoother; see [Observation003](observations/003-loss-alignment-norms.md)
for the exact event records, local windows, caption and interpretation limits.

| Metric | Run021: LR 0.001 | Run022: LR 0.0001 |
| --- | ---: | ---: |
| Initial validation loss | 11.032736 | 11.032736 |
| Final validation loss | 4.763734 | 6.232207 |
| Final validation perplexity = exp(loss) | 117.182711 | 508.877415 |
| Training loss, updates 612-711 (token-weighted) | 4.844680 | 6.395942 |

Each run covers 712 accepted updates, 728,375 sequences, 1,491,711,416 input tokens and 1,490,983,041 supervised targets without repetition. Each initial/final validation covers 339 sequences, 693,668 input tokens and 693,329 targets, including the short tail. There are only initial and final fixed-validation evaluations; intermediate training curves use changing batches.

## Geometry over the complete pass

All scalar summaries below retain update 712; adjacent statistics use 711 defined pairs. A and B have stream-specific vector units and do not represent parameter displacement.

| Stream / metric | Run021 | Run022 |
| --- | ---: | ---: |
| Raw gradient: median adjacent cosine | -0.381891 | 0.753265 |
| Raw gradient: negative adjacent pairs (%) | 84.528833 | 26.300985 |
| Raw gradient: A at update 712 | 502.558310 | 511.585644 |
| Raw gradient: B at update 712 | 125.174364 | 304.159781 |
| Raw gradient: B/A at update 712 | 0.249074 | 0.594543 |
| Raw gradient: median individual norm | 0.721823 | 0.651735 |
| Raw gradient: median cosine, updates 612-711 | -0.431767 | -0.334239 |
| Adam momentum m: median adjacent cosine | 0.922336 | 0.997485 |
| Adam momentum m: negative adjacent pairs (%) | 0.000000 | 0.000000 |
| Adam momentum m: A at update 712 | 147.331778 | 329.241160 |
| Adam momentum m: B at update 712 | 122.302779 | 301.178293 |
| Adam momentum m: B/A at update 712 | 0.830118 | 0.914765 |
| Adam momentum m: median individual norm | 0.191113 | 0.434794 |
| Adam momentum m: median cosine, updates 612-711 | 0.916120 | 0.866978 |
| Adam direction d: median adjacent cosine | 0.956919 | 0.979327 |
| Adam direction d: negative adjacent pairs (%) | 0.000000 | 0.000000 |
| Adam direction d: A at update 712 | 875413.388534 | 1226627.263963 |
| Adam direction d: B at update 712 | 574727.958083 | 699137.583450 |
| Adam direction d: B/A at update 712 | 0.656522 | 0.569967 |
| Adam direction d: median individual norm | 1157.899383 | 1571.745902 |
| Adam direction d: median cosine, updates 612-711 | 0.942365 | 0.977581 |

Run022's positive full-pass raw median is driven partly by its long early positive phase; the terminal 100-update raw median is negative in both runs. Raw A is similar across the two passes, whereas the resultant B is much smaller in Run021. Momentum smooths clipped gradients, and both momentum streams remain positively aligned even when adjacent raw gradients are negative. Adaptive adjacent alignment is higher than momentum in Run021 but lower in Run022; preconditioning does not uniformly increase adjacent cosine. Adaptive B/A is below momentum B/A in both passes. These are within-trajectory transformations, not an ablation of momentum or preconditioning.

## Temporal coverage and available profiles

The saved recovery states contain the last raw, momentum and adaptive vector at each checkpoint. The supplement uses exactly updates 100, 200, 300, 400, 500, 600 and 700, common to both runs, with all 14,067,712 unique trainable parameters. Cosines are computed directly in FP64 from saved FP32 raw/momentum and FP64 adaptive vectors. Lag 100/200/300/400/500/600 means average 6/5/4/3/2/1 sampled-pair cosines, respectively. The matrices expose every sampled pair rather than hiding its training phase in the mean.

At lag 100, mean sampled raw cosine is 0.107714 at LR 0.001 and 0.580164 at LR 0.0001. Momentum is positive in every sampled pair in both runs. Yet its longest sampled pair (updates 100 and 700) has cosine 0.708184 at LR 0.001 versus 0.486947 at LR 0.0001, so the adjacent-alignment ordering does not extend uniformly to large lags. For adaptive directions, that same pair has cosine 0.408899 versus -0.252293. The smaller-LR trajectory can have highly positive adjacent adaptive alignment while early and late sampled directions oppose. These statements describe the saved pairs; they do not infer unobserved dense lag behavior.

| Stream | Lag (updates) | Sampled pairs | Mean cosine, Run021 | Mean cosine, Run022 |
| --- | ---: | ---: | ---: | ---: |
| Raw gradient | 100 | 6 | 0.107714 | 0.580164 |
| Raw gradient | 200 | 5 | 0.040797 | 0.325046 |
| Raw gradient | 300 | 4 | 0.153748 | 0.238312 |
| Raw gradient | 400 | 3 | 0.073430 | 0.290631 |
| Raw gradient | 500 | 2 | -0.050714 | 0.135144 |
| Raw gradient | 600 | 1 | 0.107636 | 0.064587 |
| Adam momentum m | 100 | 6 | 0.887275 | 0.945485 |
| Adam momentum m | 200 | 5 | 0.873743 | 0.874162 |
| Adam momentum m | 300 | 4 | 0.866222 | 0.789356 |
| Adam momentum m | 400 | 3 | 0.830887 | 0.696655 |
| Adam momentum m | 500 | 2 | 0.793301 | 0.592518 |
| Adam momentum m | 600 | 1 | 0.708184 | 0.486947 |
| Adam direction d | 100 | 6 | 0.579827 | 0.648540 |
| Adam direction d | 200 | 5 | 0.544439 | 0.430905 |
| Adam direction d | 300 | 4 | 0.515831 | 0.247145 |
| Adam direction d | 400 | 3 | 0.488032 | 0.072800 |
| Adam direction d | 500 | 2 | 0.455388 | -0.108159 |
| Adam direction d | 600 | 1 | 0.408899 | -0.252293 |

These are exact sparse checkpoint comparisons, not dense phase-specific lag-1-to-64 profiles. Complete vector histories and temporal sketches were not recorded for these runs. Adjacent scalar cosines and cumulative vectors cannot reconstruct the missing lag profiles. Different lags average different endpoints/training phases, and a lag with one pair has no replication. No standard errors or cross-seed confidence intervals are shown. No replay or additional remote computation was launched.

## Reduction, provenance and limits

The requested main figure is 2x3, in row-major order: (a) training loss for both LRs; (b) raw adjacent cosine; (c) adaptive/momentum adjacent cosine; (d) overlay of raw/momentum A and B; (e) raw/momentum/adaptive B/A; (f) raw/momentum individual norms. Colors encode LR; line styles encode stream or A/B. Faint loss/cosine traces show unsmoothed updates. Bold loss is a trailing 25-update token-weighted mean; bold cosines are trailing 25-update medians. The main display uses full-sized updates 1-711; the partial update 712 is retained in data/statistics. Panel (c) uses a narrower y-range covering all its observed values; the norm panel uses a log scale. Undefined first cosines are gaps.

Both runs share exact initial tensors, seed 1234 data order, pinned dataset/tokenizer, model, batch, FP16 compute/FP32 Adam moments, clipping and optimizer settings. Each LR has the same 1% warmup clock followed by a constant plateau. The intended LR difference also changes AdamW's LR-scaled weight decay. m is actual uncorrected exp_avg, accumulated from clipped gradients; d is the bias-corrected preconditioned direction excluding LR/decay. No causal mediation, stability, convergence time, information storage or cross-seed generalization claim follows from these measurements. Run023 is outside this two-run analysis and remains part of the pending full cohort comparison.

Sources: each run's immutable `artifacts/attempt-01/full-pass/events.jsonl`, `summary.json`, `input_identity.json`, `verification.json`, transfer inventory and seven recovery states. Completed retrieval receipts verify all output sizes/SHA-256. This reduction rechecks the compact source hashes and recorded norms, exact clocks and scope of all sampled vectors. [Data and hashes](comparison_data.json), [sampled-pair CSV](checkpoint_pairs.csv), [figure observation](observations/001-two-run-comparison.md). Generating script: `../07_analyze_021_022.py`. Source-run artifacts and remote training are unchanged.
