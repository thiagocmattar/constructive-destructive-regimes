# Observation 001: completed three-LR results

LR 0.001 has the lowest final validation loss in this matched single-seed cohort. Both a tenfold reduction and a tenfold increase in LR give higher final loss. The large LR has much larger raw norms, more clipping and a smaller raw cumulative B/A. Positive momentum/adaptive adjacent alignment persists, despite worsening late training loss.

[Combined six metrics](../figures/metrics_by_learning_rate.pdf) | [Within-LR geometry](../figures/geometry_within_learning_rate.pdf) | [Loss, alignment and norms](../figures/loss_alignment_norms.pdf) | [Checkpoint lag profiles](../figures/checkpoint_lag_profiles.pdf) | [Checkpoint cosine matrices](../figures/checkpoint_cosine_matrices.pdf)

## Performance and geometry

| Metric | Run021: LR 0.001 | Run022: LR 0.0001 | Run023: LR 0.01 |
| --- | ---: | ---: | ---: |
| Final validation loss | 4.763734 | 6.232207 | 6.636489 |
| Final validation perplexity | 117.182711 | 508.877415 | 762.413485 |
| Training loss, updates 612-711 (token weighted) | 4.844680 | 6.395942 | 6.309059 |
| raw: median adjacent cosine | -0.381891 | 0.753265 | -0.155838 |
| raw: negative adjacent pairs (%) | 84.528833 | 26.300985 | 68.495077 |
| raw: final A | 502.558310 | 511.585644 | 1946.971153 |
| raw: final B | 125.174364 | 304.159781 | 87.307500 |
| raw: final B/A | 0.249074 | 0.594543 | 0.044843 |
| raw: median vector norm | 0.721823 | 0.651735 | 2.302105 |
| raw: last100 median adjacent cosine | -0.431767 | -0.334239 | -0.103373 |
| momentum: median adjacent cosine | 0.922336 | 0.997485 | 0.557951 |
| momentum: negative adjacent pairs (%) | 0.000000 | 0.000000 | 0.000000 |
| momentum: final A | 147.331778 | 329.241160 | 88.342299 |
| momentum: final B | 122.302779 | 301.178293 | 57.698232 |
| momentum: final B/A | 0.830118 | 0.914765 | 0.653121 |
| momentum: median vector norm | 0.191113 | 0.434794 | 0.105628 |
| momentum: last100 median adjacent cosine | 0.916120 | 0.866978 | 0.402949 |
| adaptive: median adjacent cosine | 0.956919 | 0.979327 | 0.859815 |
| adaptive: negative adjacent pairs (%) | 0.000000 | 0.000000 | 0.000000 |
| adaptive: final A | 875413.388534 | 1226627.263963 | 508570.113446 |
| adaptive: final B | 574727.958083 | 699137.583450 | 141219.958310 |
| adaptive: final B/A | 0.656522 | 0.569967 | 0.277680 |
| adaptive: median vector norm | 1157.899383 | 1571.745902 | 650.606374 |
| adaptive: last100 median adjacent cosine | 0.942365 | 0.977581 | 0.854019 |
| First negative raw pair (update) | 54.000000 | 385.000000 | 12.000000 |
| Updates exceeding raw clipping threshold | 63.000000 | 74.000000 | 517.000000 |
| Best trailing25 training loss | 4.757593 | 6.325363 | 5.708293 |
| Best trailing25 window end update | 711.000000 | 711.000000 | 520.000000 |
| Ideal-EMA reconstructed clipped B | 124.008145 | 301.835736 | 57.896101 |
| EMA endpoint correction norm | 1.914993 | 1.209670 | 0.892282 |
| Cosine between cumulative raw and momentum vectors | 0.999793 | 0.999812 | 0.857699 |

## Reading the A/B comparison

In all three runs, momentum lowers A substantially while B also decreases. The raw/momentum B curves almost overlap in Runs021/022; they separate more at the large LR. For normalized EMA m_t = 0.9*m_(t-1) + 0.1*h_t with m_0=0, the exact-arithmetic identity is sum(m_t) = sum(h_t) - 9*m_T, where h is the clipped gradient. Thus the cumulative clipped-gradient sum is preserved up to the terminal momentum correction, while averaging can reduce the sum of individual norms. The table's reconstructed clipped B uses this identity with actual stored FP32 moments; it ignores recurrence rounding and is not an independently captured clipped-gradient history. At LR 0.01, raw/momentum B separation is much larger than the endpoint correction. This is compatible with substantial clipping contributions; it does not isolate an optimization benefit of momentum.

## Captions, reductions and coverage

**Combined metrics:** 3x2 in row-major order: loss, raw cosine, momentum cosine, adaptive cosine, raw norms, momentum norms. Every plot overlays all three LRs. LR determines color/style; faint loss/cosine curves show unsmoothed records. Bold loss is a trailing 25-update token-weighted mean; bold cosine is a trailing 25-update median. Norms are unsmoothed and linear; all cosine panels share [-1.05, 1.05].

**Within-LR geometry:** 2x3, columns Run021/022/023. Upper panels compare raw/momentum A (solid) and B (dashed); lower panels compare g/m/d B/A. Stream determines color. Each first-row panel uses its own linear y-range to emphasize curve shape within its LR; the B/A row uses a shared linear y-scale. All curves are unsmoothed.

**Focused view:** two shared loss/cosine panels above three per-LR raw/momentum norm panels. Dots and vertical lines mark the first negative unsmoothed raw pair; shading marks all subsequent time, including later positive pairs. Median sign is never used for shading. Each norm panel has its own declared linear scale to retain visibility across LR ranges. The first negative pair is an event reference, not an operational onset of norm separation. For Run023 it occurs at update 12, close to warmup and momentum initialization; a full 25-update pre-event window does not exist.

All runs cover 712 accepted updates, 728,375 sequences, 1,491,711,416 input tokens and 1,490,983,041 supervised targets without repeats. The three main figures display full-sized updates 1-711; the partial final update 712 is retained in all source records and terminal statistics. Initial/final validation includes all 339 sequences, 693,668 input tokens and 693,329 targets, including the short tail. Training-loss smoothing weights supervised tokens; changing-batch training curves are not intermediate fixed-validation evaluations.

**Temporal profiles and matrices:** exact FP64 whole-model cosines using the saved previous vectors at common updates 100-700 in increments of 100, all 14,067,712 unique parameters. Lag 100/200/300/400/500/600 averages 6/5/4/3/2/1 pairs. The 3x3 matrix layout has LR rows and raw/momentum/adaptive columns, with shared [-1,1] color limits. These sparse checkpoint samples mix training phases; they do not reconstruct dense short-lag profiles. No independent-seed confidence intervals or causal conclusions are available.

At LR 0.01, sampled momentum checkpoint pairs can have negative cosine despite every adjacent momentum pair being positive. Its sampled adaptive lag-100 mean is also much smaller than its adjacent adaptive median. The following exact longest sampled pair (updates 100 and 700) illustrates the differing temporal orders:

| Stream: cosine between updates 100 and 700 | Run021: LR 0.001 | Run022: LR 0.0001 | Run023: LR 0.01 |
| --- | ---: | ---: | ---: |
| raw | 0.107636 | 0.064587 | 0.327641 |
| momentum | 0.708184 | 0.486947 | 0.112024 |
| adaptive | 0.408899 | -0.252293 | 0.055109 |

The independent-linear-scale geometry revision is generated by `../../11_plot_geometry.py` from the same verified event records. [Geometry plotting metadata](../geometry-figure.json) records source/generator hashes, per-LR y-limits and coverage.

## Provenance and limits

Sources are the completed immutable attempt-01 event logs, summaries, identities, configurations, verification files and recovery states. All retrieved files passed size/SHA-256 verification; Run023's 48 files were independently rechecked at closeout. The two previous checkpoint analyses are reused unchanged, after matching source hashes; Run023's exact matrices are newly computed. [All event records, reductions and source hashes](../comparison_data.json); [189 sampled pairs](../checkpoint_pairs.csv); [figure QA](../verification.json). Generating script: `../../10_complete_three_lr.py`, with unchanged verified helpers `../../07_analyze_021_022.py` and `../../08_plot_simplified.py`.

All conditions share exact frozen initial tensors, data order/seed 1234, full token stream, optimizer, clipping, precision, batch and runtime. Only peak LR varies within this warmup-only cohort; LR also scales AdamW decay displacement. g is raw pre-clipping; m is actual uncorrected first moment of clipped gradients; d is bias-corrected/preconditioned, excluding LR/decay. A/B values have stream-specific units and are not actual parameter displacement. One matched seed does not support cross-seed generalization, a stability criterion, noise variance, information retention or a causal mediation claim. Previous interim PDFs and all executed scientific inputs are preserved; no new training, replay or cloud spending was launched.
