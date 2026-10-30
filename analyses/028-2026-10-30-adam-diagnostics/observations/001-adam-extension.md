# Evidence for the Adam extension

October 30, 2026. Existing completed trajectories only. This observation
documents the two tables and numerical prose in Section VI; it does not
introduce a new learning-rate experiment or infer a validated controller.

## Seven-rate stream comparison: Table II

**Sources.** Runs021--027 and the hash-identified event arrays in
[Analysis009 comparison_data.json](../../009-2026-10-09-intermediate-lr/complete-seven-lr/comparison_data.json).
The original three rates are from
[Analysis008](../../008-2026-10-08-no-decay-momentum/complete-three-lr/README.md).
The [seven-rate observation](../../009-2026-10-09-intermediate-lr/complete-seven-lr/observations/004-implications-for-manuscript.md)
already distinguishes local alignment, cumulative interaction and loss.

**Comparison.** One shared Pythia-14M initialization and data order, seed 1234;
peak LR varies with a matched warmup-only schedule. Other scientific inputs
are matched. Rates 1e-4, 1e-3 and 1e-2 ran on DGX Spark/GB10; the four
intermediate rates ran on RTX 4090. Table II groups these platforms explicitly.
Cross-platform effects cannot be attributed solely to LR. LR also scales
AdamW's decay displacement.

**Coverage.** Each run processes 728,375 packed sequences, 1,491,711,416 input
tokens and 1,490,983,041 next-token targets in 712 accepted updates. There are
711 full batches of 1,024 sequences and one final batch of 311, with microbatch
four and context 2,048. Warmup spans the first 1% of updates; the first LR is
zero while moments advance. The recipe uses AdamW beta=(0.9,0.95), epsilon1e-8,
clip norm1, decay0.1 excluding biases/LayerNorm, FP16 dynamic loss scaling with
FP32 parameters/moments and FP64 diagnostic reductions. All unique trainable
parameters are included. Validation covers all 500 documents, 339 packed
sequences and 693,329 targets, including the tail; loss is token weighted.
Frozen source/config inventories in the owning runs retain model, tokenizer,
dataset/cache and ordering identities. The models are initialized from scratch,
not pretrained Pythia checkpoints.

**Vector definitions.** Raw gradients are unscaled accumulated gradients before
clipping. Momentum is the optimizer's actual uncorrected `exp_avg`, updated
from clipped gradients. Adaptive directions include first/second-moment bias
correction and coordinate scaling, but exclude LR and decoupled decay. These
three streams are not three independent optimizer trajectories. The first
undefined adjacent cosine is omitted; all subsequent cosines are defined.

**Reduction and caption.** Table II reports the median unsmoothed consecutive
cosine over updates612--711 (100 full-batch observations) for each stream and
the final full-validation loss. It uses neither temporal smoothing nor seed
averaging. Values are rounded to three decimals only at display; LR is rounded
to three significant figures. The caption states the window, streams, single
initialization and hardware grouping. The last partial update is excluded from
the late median but retained in final validation and whole-run counts.

**Supported findings.** All seven late raw medians are negative, while moment
and adaptive medians are positive. All 711 defined adjacent momentum pairs
per run are positive. LR0.01 has late median momentum cosine0.402949, raw
norm5.565900 and moment norm0.096840; 517/712 updates exceed the clipping
threshold. Those norm contrasts do not identify causal propagation into
momentum. Within RTX 4090, LR0.004641589 has lower momentum alignment than
LR0.002154435 (0.852282 versus0.956273), with lower final validation loss
(4.318091 versus4.410370).

**Uncertainty and limits.** One initialization; serial observations are not
independent replicates and no confidence interval is estimated. The selected
late-window medians describe these trajectories. There is no universal LR
optimum, cosine threshold or stability boundary. Input-versus-previous-moment
alignment and actual-update cumulative retention are not measured here.

## Matched schedule intervention: Table III

**Sources.** Runs011--016, their parents005/007 and
[Analysis004 results.json](../../004-2026-10-04-lr-schedule-intervention/results.json).
The original [observation](../../004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md)
specifies parent-state hashes, replay qualification, coverage and sensitivity.

**Comparison.** Pythia-14M and 70M each use one seed1235 checkpoint after
update356. Branches match weights, optimizer state/clock, scaler, RNG, data
position and future batches. The interventions are original cosine decay,
holding LR0.0005179103055, and advancing the decay clock twice as fast.
Original/accelerated decay reach floor0.0001 at updates575/466. All six new
continuations run on RTX 5090, through update712, with the same optimizer,
precision, batch and complete-validation recipe. Prefix plus continuation
covers the full training cache. Each continuation processes 363,831 sequences,
745,125,304 input tokens and 744,761,473 targets, including the partial tail.

**Reduction and caption.** The raw gradient at update357 is common before the
changed parameter update. Table III therefore uses affected adjacent pairs
358--712: 355 equally weighted, unsmoothed observations per branch. The final
partial batch is retained. Each cell pair gives median raw cosine and final
token-weighted validation loss, rounded to three decimals. The caption states
the matched checkpoint design, affected window and one initialization per size.

**Supported findings.** Faster decay increases median raw cosine at both sizes;
holding LR lowers it. Yet held LR has the lowest terminal validation loss in
both sizes, and accelerated decay the highest. Earlier positive smoothed
alignment follows the original Analysis004 criterion: the first 31 consecutive
positive centered31 means in the eligible358--711 window. Its first centers
are467/402 for original/accelerated14M and538/437 for70M. Both decaying70M
branches subsequently return to a negative smoothed mean. This retrospective
criterion is not an online trigger; the manuscript reports earlier recovery
qualitatively and does not introduce its extra notation.

**Uncertainty and limits.** One initialization per size, one matched branch
state, no across-seed confidence intervals. The intervention identifies a
schedule effect conditional on those states; changing LR also changes the
AdamW decay displacement. It does not isolate a curvature change, prove that
alignment mediates improvement, or establish the predictive value of a
diagnostic-driven schedule. The original partial-batch sensitivity preserves
the directions of the reported contrasts.

## Verification and presentation

[01_build_evidence.py](../01_build_evidence.py) checks the 13 source event hashes,
reconstructs the scalar reductions, and writes both TeX tables.
[02_verify_manuscript.py](../02_verify_manuscript.py) checks all 40 displayed
cosine/loss values and the numerical prose, tests the stated geometric
qualifications on small deterministic examples, and checks the compiled
manuscript. Hashes and results are in [evidence.json](../evidence.json) and
[verification.json](../verification.json).

The tables are adjacent, legible at one-column IEEE size, and share three-decimal
precision. There are no smoothing curves or significance marks. They answer
two different questions: what information the optimizer transformations retain,
and how that information responds to a matched LR intervention. Existing
figures and raw evidence remain at their original paths; no new publication
figure is necessary for this compact extension.
