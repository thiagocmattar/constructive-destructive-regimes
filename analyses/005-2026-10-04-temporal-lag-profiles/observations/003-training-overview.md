# Observation003: training loss, eta-kappa proxy, rho and raw norms

The requested overview reuses the three completed Analysis005 measurement replays:
Pythia-14M/31M/70M, Runs017/018/019, seed 1235. It adds a four-row, three-column
[figure](../figures/training_overview.pdf). No training, replay, cloud resource,
new seed, or manuscript change is needed.

## Sources and retained state

| Model | Source run | Retained artifact directory |
| --- | --- | --- |
| 14M | [Run017](../../../runs/017-2026-10-04-pythia14m-temporal-sketch-seed1235/README.md) | `artifacts/attempt-01/full-pass/` |
| 31M | [Run018](../../../runs/018-2026-10-04-pythia31m-temporal-sketch-seed1235/README.md) | `artifacts/attempt-02/full-pass/` |
| 70M | [Run019](../../../runs/019-2026-10-04-pythia70m-temporal-sketch-seed1235/README.md) | `artifacts/attempt-02/full-pass/` |

Each directory retains all 712 update events, raw-gradient CountSketch histories,
exact adjacent dot products, and initial/midpoint (update 356)/final model and
recovery checkpoints. These are the actual Analysis005 trajectories; their
earlier reference Runs005/006/007 are numerically different.

The generator checks the sizes and SHA-256 of all consumed scalar/config/identity/
exact-pair sources against both the accepted retrieval inventory and the original
Analysis005 reduction. It verifies the accepted receipt hashes. All nine retained
checkpoint states have model/recovery files, with sizes matching the inventories;
checkpoint contents are not rehashed or loaded for this scalar-only analysis.
The [manifest](../training_overview_manifest.json) records paths, sizes, hashes,
the actual plotted-array checks and visual review. The six original PDFs and
original results/data/acceptance record are unchanged.

## Comparison and measurement contract

Model architecture/size varies. Within seed 1235, dataset/tokenizer revisions,
cache bytes, ordering, total tokens, accumulation boundaries, optimizer/LR schedule,
loss reduction, clipping, precision and diagnostic capture match across sizes.
Architecture-specific initialized tensors differ; this is one seed per size.
Each run processes 1,491,711,416 input tokens in a full MiniPile pass, with packed
length 2048 and effective batch 1024 sequences, accumulated from microbatches of 4.
The final update contains 311 sequences, including the short token tail.

Loss is each update's unsmoothed mean causal next-token cross-entropy, weighted
by that update's supervised-token count. It is a changing minibatch objective,
not a full-dataset objective at each saved state. Each point uses parameters
before its associated optimizer update. Plotting uses the saved one-based update
clock 1..712; the underlying gradient_step clock is 0..711. Initial/final validation
remains available with all 500 validation documents and 693,668 input tokens;
validation is not plotted as if it were a dense training-loss trace.

`g_t` is the entire unique trainable-parameter gradient after accumulation and
FP16 loss unscaling, before clipping, AdamW transformation, LR or weight decay.
Stored raw vectors are FP32, with FP64 diagnostic reductions. Norm and cosine
are exact recorded scalars, not CountSketch estimates:

```text
rho_t = dot(g_(t-1), g_t) / (norm(g_(t-1)) * norm(g_t))
q_t = eta * kappa_hat_t = 1 - rho_t * norm(g_t) / norm(g_(t-1))
    = 1 - dot(g_(t-1), g_t) / norm(g_(t-1))^2
```

This follows the gradient-only expression in
[Analysis024](../../024-2026-10-17-eos-gd-lr-calibration/README.md).
No LR division or additional LR multiplication is performed: q is already the
dimensionless product. The underlying schedule varies and starts at zero LR;
in particular, q at update 2 remains an algebraic gradient-change quantity even
though the preceding update used zero LR. No finite kappa estimate is assigned
to that zero-LR step. The minibatch-noise correction is unmeasured here and is
not included; this does not assert that the actual stochastic noise is zero.

The first rho/q pair is undefined. Either zero adjacent norm would also make
the literal expression undefined; no denominator floor, clipping or artificial
zero is inserted. All actual norms are positive, leaving 711 defined pairs per
model. Every update is included, without smoothing, thinning, tail exclusion or
weighting between updates. Exported missing values are blank cells.

## Results and figure caption

| Model | First negative rho (update) | Negative pairs / 711 | q range | Final batch training loss | Final validation loss |
| --- | --- | --- | --- | --- | --- |
| 14M | 63 | 356 (50.07%) | -0.1802 to 2.8332 | 5.2225 | 5.1822 |
| 31M | 26 | 397 (55.84%) | -0.0199 to 2.1300 | 4.5846 | 4.6249 |
| 70M | 14 | 469 (65.96%) | -0.0198 to 2.5296 | 3.9311 | 4.0099 |

Larger models reach lower final loss and have more negative adjacent pairs in
these trajectories. q exceeds 1 exactly when rho is negative: these panels encode
an algebraic relationship, not two independent pieces of evidence. The large
terminal raw-norm changes, and the 14M terminal q spike, coincide with the smaller
last batch; its diamond marker prevents treating it as another full batch.

**Caption.** Training dynamics of Pythia-14M, 31M and 70M on one full MiniPile
pass, seed 1235 (Runs017-019). Rows show unsmoothed training loss, dimensionless
gradient-change proxy q=eta*kappa_hat, raw adjacent-gradient cosine rho, and raw
gradient L2 norm. Columns use common Y limits within each row; only the norm row
uses log Y. All 712 updates are retained, with 711 defined adjacent pairs per
model. Diamonds identify the partial final batch (311 versus 1024 sequences).
Dashed lines mark q=1 and rho=0. Raw-gradient diagnostics precede clipping and
AdamW. Changing minibatches and the AdamW displacement prevent reading q as the
full-GD curvature quantity; neither reference line is a stability criterion.

## Verification, uncertainty and reproduction

All 2,133 identities are checked independently against the stored exact lag-one
dot products, including norm/index placement; q>1 iff rho<0 holds for every pair.
Checks also cover contiguous updates, data/order and protocol matching, token/tail
coverage, zero overflow retries, undefined/zero-norm conventions, a quadratic-GD
contraction case, CSV roundtrip, and all 8,538 finite plotted values. The PDF is
rendered with Poppler and visually inspected before acceptance.

This is a descriptive one-seed comparison, without seed uncertainty bands or
causal, generalization, curvature-estimation or stability conclusions. Exact
scalar arithmetic avoids sketch approximation error, but retains the original
training precision and measurement-replay sensitivity. Adjacent gradients use
different minibatches, and AdamW includes clipping, moments, decay and a schedule.
No Hessian measurement or momentum/update direction is substituted for raw g.

Generate with `python analyses/005-2026-10-04-temporal-lag-profiles/12_plot_training_overview.py`.
The script writes the PDF, [plot data](../training_overview_data.csv),
[results](../training_overview_results.json) and manifest. Render the PDF with
`pdftoppm -png -r 115` to `scratch/training_overview`, inspect the page, then run
the generator with `--accept-reviewed` to record visual acceptance. Raw training
artifacts and preview images remain outside Git. No new run/analysis ID is used.
