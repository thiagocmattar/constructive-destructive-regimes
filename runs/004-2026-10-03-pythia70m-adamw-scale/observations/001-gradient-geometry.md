# Observation 001: pythia-70m raw and adaptive geometry

Question: measure the same raw-gradient and adaptive Adam-direction diagnostics as
Run002, to support the user-authorized scale comparison (H002).

Sources: attempt 01/full-pass events.jsonl, summary.json, input_identity.json,
environment.json, resolved_config.json, verification.json, and recovery checkpoints.
Strict byte/SHA acceptance and independent local scientific verification are in
../prelaunch/attempt1-retrieval.json. Source/script hashes, all 712 measurement rows,
displayed-prefix statistics, and fixed quarters are in ../figures/figure_data.json.

Generating command: python runs/004-2026-10-03-pythia70m-adamw-scale/04_analyze.py
Entry point calls ../../../tools/plot_scale_run.py. Publication PDFs: raw_gradient_evolution.pdf,
adaptive_evolution.pdf, raw_adaptive_comparison.pdf. Scratch PNGs are ignored.

Coverage: all 712 accepted updates, 728,375 sequences,
1,491,711,416 input tokens and 1,490,983,041
supervised targets, each once. Main plots show updates1–711, all full batches1024;
the final 311-sequence update is omitted from display only. All table/statistics
below retain it. Cosine begins at 2; undefined first value is a gap.

Raw g is unscaled after token-weighted accumulation, before global clipping.
Adaptive d is evaluated in FP64 from actual updated FP32 moments of clipped g:
bias-corrected m / (sqrt(bias-corrected v) + epsilon). It excludes LR and decay.
All unique trainable parameters, identical ordering across streams, FP64 cumulative
vectors/reductions; previous g FP32, previous d FP64. A=sum norms, B=norm sum,
R=B/A. Zero-denominator values are null with reason. The zero-LR first step still
advances moments, and its direction is included.

| Full pass | Raw g | Adaptive d |
| --- | ---: | ---: |
| Median adjacent cosine | -0.160566 | 0.931795 |
| Negative / defined cosine pairs | 481/711 | 0/711 |
| Final A | 304.620311 | 1563927.224143 |
| Final B | 94.243503 | 953113.159846 |
| Final B/A | 0.309380 | 0.609436 |
| Median vector norm | 0.394048 | 2132.926334 |

Median paired adaptive/raw norm ratio:
5382.225966.

| Updates | Raw median cosine | Adaptive median cosine | Raw end R | Adaptive end R |
| --- | ---: | ---: | ---: | ---: |
| 1–178 | -0.114392 | 0.942876 | 0.297327 | 0.576002 |
| 179–356 | -0.282898 | 0.922653 | 0.274176 | 0.583830 |
| 357–534 | -0.204856 | 0.928971 | 0.285572 | 0.591011 |
| 535–712 | 0.010837 | 0.937289 | 0.309380 | 0.609436 |

Validation loss: 11.063131 →
4.063201, all 500 fixed validation documents, including
the short tail, weighted by supervised targets. One random initialization seed 1234.

Captions: each single-vector 2×2 figure shows adjacent cosine (faint unsmoothed
values plus trailing 25-update median), A/B, R, and individual L2 norm. The paired
3×2 figure shows cosines and R on common axes, raw/adaptive A/B separately, norms
and paired norm ratio on log scales. Updates1–711 only, first cosine undefined.
No uncertainty band; trajectory quarters are fixed, not inferred regimes.

Limits: adaptive directions are not actual parameter displacements. No stability,
causal benefit, per-sample decomposition, or generalization across seeds follows
from these norms/alignments. B/A<1 can result from non-collinearity even when all
adjacent cosines are positive. Absolute scales differ after preconditioning.
Cross-size conclusions are owned by Analysis001; no scaling law is established
by this run alone. Frozen training inputs and original manuscript assets are preserved.
