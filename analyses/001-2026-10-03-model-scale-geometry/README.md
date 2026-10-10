# Analysis 001: Raw and adaptive geometry versus Pythia scale

Status: complete; both new full passes retrieved/verified and workers removed.
Hypothesis H002 has descriptive scale trends; predictability remains inconclusive.

Full-pass median cosine and final B/A decrease across 14M, 31M, 70M for both
raw gradients and adaptive Adam directions. Raw A/B decrease; adaptive A/B
increase. Median raw norm and phase-specific raw cosines are not uniformly
monotonic. See [the observation](observations/001-scale-comparison.md),
[time-series figure](figures/geometry_by_scale.pdf),
[A/B figure](figures/cumulative_by_scale.pdf),
[size summary](figures/scale_summary.pdf), and
[norm comparison](figures/norm_comparison_by_scale.pdf).

Reproduce from the repository root:
`python analyses/001-2026-10-03-model-scale-geometry/01_compare.py`, then
`python analyses/001-2026-10-03-model-scale-geometry/02_record.py`.
Exact source/config hashes and aggregates: `results.json`; all 2,136 per-step
reductions: `figure_data.csv`; PDF/source hashes: `record_generation.json`.
All 10 new run/analysis PDFs were rendered and their text bounds verified.
The [final review](final-review.json) records artifact checks, test results,
manuscript preservation, worker cleanup, and the provisional cost estimate.
Combined estimated GPU/storage cost is approximately $3.33, provisional.
No task-owned cloud storage remains; unrelated account resources were preserved.

The user proposes predictable scale behavior in adjacent cosine and cumulative
A, B, and B/A, and requests both raw gradients and adaptive AdamW directions.
No direction or parametric scaling law was specified. This comparison is
descriptive, not a pre-specified test of a slope or functional form.

Inputs: Run002 attempt02 (14M), Run003 attempt01 (31M), Run004 attempt01 (70M),
all full MiniPile passes from random SmallInit/WangInit with seed 1234. Match
tokenizer/cache bytes and sequence order, 712 updates, effective batch 1024,
microbatch 4, token-weighted loss, LR, AdamW groups/betas/epsilon/clip, FP16
dynamic scaling, validation coverage, and diagnostic boundaries. Architecture
varies: six layers with width 128/256/512 and heads 4/8/8. Initialization recipe
is matched; tensor values/shapes are distinct. Actual counts are used on axes.

For x=g (unscaled preclip batch gradient) or x=d (bias-corrected adaptive Adam
direction from actual poststep moments), A_t=sum_{i<=t}||x_i||,
B_t=||sum_{i<=t}x_i||, R_t=B_t/A_t, and c_t=cos(x_{t-1},x_t).
d excludes LR and weight decay, and uses moments of the clipped gradient.
The requested per-step norm comparison is ||d_t||/||g_t||.

The 14M source is the paired Run002 trajectory, not Run001's separate replay.
Full terminal statistics retain all 712 updates and the final 311-sequence batch.
Main time-series figures display updates 1–711, excluding the final partial batch
at the user's request. Adjacent cosines start at update 2. Rolling medians are
labeled and unsmoothed data retained. Statistics at 711 distinguish
display selection from full-pass conclusions.

One trajectory per size has no seed uncertainty estimate. Steps are correlated,
not independent replicates. Three sizes cannot establish out-of-sample
predictability or a universal scaling law. Width and head configuration co-vary
with parameter count. Absolute norms and A/B depend on dimensionality;
dimensionless cosine/R are the primary geometric comparisons. No loss decrease,
generalization, or stability conclusion follows from alignment alone. Validation
loss is measured independently over the complete fixed validation cache.

Outputs: retained reduced data and source hashes, publication PDFs, observation
with captions/coverage/caveats, and reproducible generating code.
