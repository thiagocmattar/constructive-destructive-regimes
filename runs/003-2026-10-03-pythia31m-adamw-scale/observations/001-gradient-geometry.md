# Observation 001: pythia-31m raw and adaptive geometry

Question: measure the same raw-gradient and adaptive Adam-direction diagnostics as
Run002, to support the user-authorized scale comparison (H002).

Sources: attempt 01/full-pass events.jsonl, summary.json, input_identity.json,
environment.json, resolved_config.json, verification.json, and recovery checkpoints.
Strict byte/SHA acceptance and independent local scientific verification are in
../prelaunch/attempt1-retrieval.json. Source/script hashes, all 712 measurement rows,
displayed-prefix statistics, and fixed quarters are in ../figures/figure_data.json.

Generating command: python runs/003-2026-10-03-pythia31m-adamw-scale/04_analyze.py
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
| Median adjacent cosine | -0.086252 | 0.952865 |
| Negative / defined cosine pairs | 405/711 | 0/711 |
| Final A | 305.977000 | 1207840.602857 |
| Final B | 101.202350 | 803888.443751 |
| Final B/A | 0.330751 | 0.665558 |
| Median vector norm | 0.386883 | 1627.648568 |

Median paired adaptive/raw norm ratio:
4209.503357.

| Updates | Raw median cosine | Adaptive median cosine | Raw end R | Adaptive end R |
| --- | ---: | ---: | ---: | ---: |
| 1–178 | -0.043643 | 0.965668 | 0.356766 | 0.597099 |
| 179–356 | -0.321731 | 0.951089 | 0.290246 | 0.638485 |
| 357–534 | -0.106262 | 0.950196 | 0.304165 | 0.654129 |
| 535–712 | 0.163892 | 0.953991 | 0.330751 | 0.665558 |

Validation loss: 11.002269 →
4.611768, all 500 fixed validation documents, including
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
