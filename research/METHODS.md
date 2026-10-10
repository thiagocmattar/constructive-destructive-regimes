# Experimental methods

The first selected method is a dense Pythia-14M AdamW full MiniPile pass, with
raw-gradient diagnostics. Its concrete recipe is in Run001; SGD is deferred.
Other experiments and interventions remain open.

Run002 adds actual-moment adaptive Adam directions alongside raw gradients.
Runs 003/004 match its tokens, updates, batch, AdamW/LR recipe and seed while
varying Pythia width/head configuration to 31M/70M. Initialization recipe is
matched; tensor shapes/values differ. [Analysis001](../analyses/001-2026-10-03-model-scale-geometry/README.md)
owns the full three-size comparison. One seed and three sizes supply descriptive
evidence, not a predictive scaling law. Other metrics remain undecided.

Each concrete design must name the question, comparison, matched controls, model
and initialization, data order, optimizer details, learning-rate schedule, batch
decomposition, token/step budget, seeds, precision, and checkpoint policy.
Choose whether the comparison is matched in tokens, updates, wall time, or another
declared quantity; changing batch size can make these different comparisons.

Before scientific code, resolve `DEFINITIONS.md` and create explicit metric
contracts using `METRICS.md`. Validate any mathematical decomposition against
synthetic cases and actual recorded updates, including normalization and indexing.
Keep approximation error visible rather than assuming it away.

Diagnostic sampling and storage must be planned alongside computation. Per-sample
gradients, temporal histories, and fixed-parameter probe gradients have different
costs and meanings. Cadence changes the lag represented by consecutive records.
Quantify overhead on the proposed workload before extrapolating duration.

Do not import sparsity pressure, gating, orthogonal L1, or custom runtime kernels
unless a future question explicitly requires them. The new experiments should
answer this paper's question with the smallest suitable implementation.

Analysis002 completed six additional full passes, yielding three seeds per size.
[Seed replication](../analyses/002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)
reports raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks. Data order is matched across sizes within seed and
varies with initialization between seeds. Seed1234 used RTX4090, new blocks RTX5090;
numerical hardware effects cannot be separated from seed effects across blocks.
All per-run values, reversals, A/B/norms and validation are retained; no predictive
law, stability result or causal learning benefit is established.

[Analysis008](../analyses/008-2026-10-08-no-decay-momentum/README.md) owns three
user-authorized Pythia-14M full passes with warmup followed by constant LR:
1e-3, 1e-4 and 1e-2. Runs 021–023 share Run002's exact initial tensors, seed 1234
data order, complete cache and optimizer/batch recipe on the same DGX Spark.
Every accepted update captures raw gradient, actual first moment m and adaptive
direction geometry. LR also scales AdamW decay displacement. This is a matched
one-seed intervention; historical Run002 decay curves have a hardware/kernel
difference, and no cross-seed uncertainty or stability inference is assumed.
