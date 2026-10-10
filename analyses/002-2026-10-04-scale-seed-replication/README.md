# Analysis 002: Does the size pattern survive new seeds?

Status: all six new full passes completed; archives accepted; all six workers and volumes deleted.
H002 is examined across three seeds per size: 1234, 1235, and 1236.

The user suspects a single-run effect in Analysis001 and requests two more seeds
at 14M, 31M, and 70M, including both raw gradients and adaptive Adam directions.
New runs 005/006/007 pair seed1235 across sizes; 008/009/010 pair seed1236.
Run002/003/004 provide the existing seed1234 block. Model initialization and
training-data permutation both change between seed blocks; data order is matched
within a seed across sizes. This controls batch exposure for each size comparison
while testing variability across training realizations. It does not separate
initialization variability from order variability.

Use the complete matched full-pass recipe and metric definitions from
[Analysis001](../001-2026-10-03-model-scale-geometry/README.md). Raw gradients are
captured before clipping; adaptive directions use actual AdamW moments of clipped
gradients, before learning-rate scaling and weight decay. The same exact data
cache, tokenizer, coverage, 712 updates, loss weighting, optimizer, schedule,
precision, checkpoint policy, and parameter scope apply. Architectures vary
within the original width/head recipe. All new workers use RTX5090; seed1234 used
RTX4090. Hardware/kernel numerical differences are retained as a limitation.

## Comparison decided before the new full passes

The pilot ordering is a descriptive expectation for replication, not a universal
law. Primary summaries, separately for raw/adaptive vectors:

- Median adjacent cosine over all 711 defined pairs decreases 14M > 31M > 70M.
- Terminal B/A at update712 decreases 14M > 31M > 70M.

Report strict ordering for each seed block and paired differences 31M−14M and
70M−31M. Report every per-run estimate, cross-seed mean, sample SD and min/max;
plot seed points and individual curves. Steps within a trajectory are correlated
and are never treated as independent seed replicates. A reversal is retained.
No significance claim or fitted predictive scaling law is planned with n=3.

Secondary summaries: terminal A and B, individual norm medians, paired adaptive/raw
norm ratio medians, negative-cosine fractions, full validation loss, phase-specific
cosine medians, and minimum B/A. Compare the pilot directions (raw A/B decrease;
adaptive A/B and norms increase) without hiding nonmonotone results. No causal
optimization or stability claim follows from geometric diagnostics alone.

Retain every raw update and final partial batch. Main evolution displays omit
update712 at the user's earlier request; full-pass statistics include it. Also
report update711 sensitivity. Smooth only labeled descriptive cosine curves;
seed envelopes represent between-run spread, not a confidence interval.

## Execution and closeout

Six Secure Cloud RTX5090s, live $0.99 per GPU-hour on 2026-10-04; at most four
hours per worker and $25 for the cohort including storage and bounded retries.
Initial ETC 2–3 hours, qualified from previous RTX4090 timings and pending actual
RTX5090 calibration. 60GB container/40GB persistent storage per Pod, no network
volumes. All workers have recorded ownership, deadlines and hidden local guards.
Processes detach and save progress/artifacts; read-only monitoring is bounded.
Calibration and production fused-Adam audits precede the full passes. Download
and verify file bytes/SHA and independent scientific checks before removing each
worker. Retain results/checkpoints locally and remove task-owned cloud storage.

This analysis will own reduced CSV/JSON, publication PDF figures, observations
with sources/captions/coverage/uncertainty, and reproducible generating scripts.
No manuscript edits or additional experiments are included.

## Completed results

Strict pilot ordering (14M > 31M > 70M): raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks.
The main full-pass ordering survives both additional seeds for both streams. Adaptive A/B/norms
increase in all seeds; raw B decreases in all seeds, while raw A has a small 31M-to-70M
reversal in seed1235. Raw phase-specific cosine ordering is not uniform. This is a
three-seed reproducibility check, not a fitted predictive scaling law.

[Observation001](observations/001-seed-replication.md) reports every seed, means/sample SD, A/B/norms, phase summaries and limits.
[Geometry](figures/geometry_across_seeds.pdf), [size summaries](figures/scale_summary_across_seeds.pdf),
[cumulative A/B](figures/cumulative_across_seeds.pdf), and [norm ratios](figures/norm_ratio_across_seeds.pdf) are publication PDFs.
All 6,408 update rows remain in [figure_data.csv](figure_data.csv); exact lineage is in [results.json](results.json).

Six new runs cost an estimated $10.11 including running storage, provisional until provider billing settles.
No task-owned cloud storage remains. [Final review](final-review.json) records verification and preservation.
