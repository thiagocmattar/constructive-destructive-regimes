# Recommended experiment on late learning rate decay

Test whether changing LR decay after a common training prefix shifts subsequent
raw-gradient cancellation and loss. Branch the retained midpoint checkpoints for
Pythia14M and Pythia70M, using all three existing seeds, into original decay,
held LR, and accelerated decay. This focuses the experiment on the late recovery
seen in Analysis003. The [completed replication audit](observations/003-replicated-conclusions.md)
supports retaining the restricted conclusion about local cancellation and loss.

This is a recommendation for H004, with proposed choices below. No new run ID,
cloud resource, schedule implementation, or execution has been created.

## Comparison and schedules

Use the checkpoint saved after optimizer update 356, before the observed pilot
recovery. Compare updates 357-712, consuming the same remaining 363,831 packed
sequences within each parent checkpoint, including the final partial batch and
the short packed sequence. Match the exact remaining input and target tokens
within each parent; record their counts from the data order.

Let eta_ref(u) be the frozen, executed GPT-NeoX pre-step recipe from
`runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/artifacts/attempt-02/source/baseline/recipe.py`.
Its original budget remains 712, peak 0.001, minimum 0.0001, and warmup fraction
0.01. Do not restart the schedule with a 356-update budget. The midpoint rate is
eta_ref(356) = 0.0005179103055038871.

| Proposed branch | LR for updates357-712 | First floor update | Purpose |
| --- | --- | ---: | --- |
| Original decay | eta_ref(u) | 575 | Replay control on the same execution stack |
| Held LR | eta_ref(356) | Never | Remove further annealing after the shared prefix |
| Accelerated decay | `eta_ref(min(712, 356 + 2*(u - 356)))` | 466 | Advance the decay clock by a factor of two |

The accelerated rule follows the same curve from the same junction, retaining
the original floor. The held branch means **no further decay after update356**.
It does not test constant LR from initialization; an experiment with warmup
followed by constant peak LR would address a different, whole-training question.

Two sizes x three seeds x three branches give **18 continuations**, each 356
updates. Their 6408 total optimizer updates equal nine original full passes in
update count. This is not a runtime or cost estimate: model sizes, validation,
resumption, checkpointing, diagnostic overhead, and transfer have different costs.
The 31M model can be added later if the endpoint comparison warrants interpolation.

## Matched factors and provenance

Within each size/seed, restore identical model tensors, AdamW moments and step
counters, dynamic loss scaler, Python/NumPy/Torch/CUDA random states, cached token
bytes, permutation and data position, and raw/adaptive diagnostic state. Match
batching, precision, dropout, clipping, loss weighting, parameter scope, and all
optimizer settings; only the future LR rule varies. Use one hardware/software
stack for all conditions, and balance or randomize branch execution order within
seed to avoid making each schedule coincide with a different resource cohort.

Retain the existing recipe: effective batch of 1024 sequences, microbatch of four,
packed length of 2048 tokens, FP32 parameters with FP16 dynamic scaling, AdamW
betas (0.9, 0.95), epsilon 1e-8, global clipping 1.0, and weight decay 0.1 with
bias/layer-norm exclusions. Dataset,
tokenizer and architecture revisions, token-cache hashes, initialization identity,
parameter-name order, and source config come from each verified parent. Each new
branch must record parent checkpoint/config hashes and its new LR rule and code
hashes, without modifying the parent artifacts.

The six required midpoint directories contain model tensors and `recovery.pt`;
their presence and sizes are inventoried in `replication_check.json`. The current
trainer saves recovery state but has no implemented resume entry point. Before
scientific execution, implement a minimal run-local continuation loader and verify
restoration, skipped-data indexing, and uninterrupted versus resumed mechanics.
Compare the original-decay continuation with the archived trajectory on the
chosen stack, recording numerical deviations instead of promising bitwise replay
across GPU types. The new replay branch remains the contemporaneous control.

## Measurements and descriptive analysis

Capture raw accumulated gradients at every optimizer boundary after unscaling,
before clipping or AdamW. Use all unique trainable parameters, FP32 captured
vectors, FP64 dot/norm reductions, and null with an explicit reason for a zero
denominator. Retain the existing adaptive direction diagnostics as secondary
continuity checks; they exclude LR and decay and are not actual parameter updates.

The primary descriptive contrast is the within-parent change versus original
decay in mean max(-rho, 0) over pairs358-712. Report every seed's difference for
held and accelerated decay, separately by size, plus cross-seed mean and range.
Also report median cosine, negative fraction F, conditional magnitude M, raw
norms, and token-weighted training loss. Do not count the 355 correlated pairs as
355 independent replications or select the branch with the highest cosine as
the best optimizer.

There is an indexing check with scientific consequences: gradient g357 and its
cosine with g356 are evaluated before the first changed parameter update. They
should match across branches on the same stack. LR intervention at update357
can first affect g358. Preserve the global optimizer/bias-correction clock even
when creating diagnostic summaries for the continuation alone.

Keep cumulative A/B/R over the inherited whole trajectory. If reporting A/B/R
for the continuation alone, use a separately initialized diagnostic accumulator
and label its horizon; resetting inherited geometry must never reset Adam's
bias-correction step. Account for any additional vector buffer in memory and
overhead calibration before execution.

Use fixed reporting intervals357-445,446-534,535-623,624-712, with update357
identified as a common pre-intervention measurement. Evaluate the same complete
500-document validation split at356,445,534,623,712, including its short packed
tail and all 693329 supervised targets. The junction evaluation is common to all
branches from that parent. Keep token-weighted loss separate from equal-step
geometric summaries; validation loss measures held-out fit under this coverage,
without assigning a mechanism to cosine sign.
Intermediate validation must preserve training RNG state and restore training
mode so that evaluation cadence does not alter the continuation trajectory.

Make a 1x2 panel per size, showing loss and raw cosine, with individual seed
curves and a thin cross-seed average for each schedule. Use the existing centered
31-update mean, with 15/61-update sensitivity, and mark the junction and floor
times. Draw the LR curves in a separate schedule panel. Preserve update712 in
data and terminal summaries, while omitting it from main evolution plots as in
the earlier analyses. Smoothed curves are for display, not primary aggregation.

As an exploratory timing summary, retain observation002's first run of 31
positive centered means, using only complete post-intervention windows and
reporting no event when the criterion is not reached. Report smoothing
sensitivity and later reversals. This marker is retrospective and does not
define permanent recovery or a phase transition.

## What the outcomes would support

If holding LR prolongs negative alignment while faster decay advances positive
alignment consistently within seeds, that supports a contribution of late LR
decay to the measured trajectory geometry, conditional on the trained prefix.
If the contrast differs between 14M and 70M, report the size-dependent response
without inferring a dimension-only mechanism. If differences are small or
inconsistent, the tested variants provide limited evidence for the proposed
explanation; three seeds cannot establish that schedules have no effect.

Compare alignment changes with validation and training loss separately. Earlier
positive alignment may coexist with better, equal, or worse loss. A schedule
intervention can support an LR effect on geometry without showing that geometry
mediates the loss effect. AdamW also multiplies weight decay by LR, so changing
the LR changes both the adaptive displacement and per-step shrinkage. The
comparison identifies the effect of this LR recipe with weight decay held at
0.1; it does not isolate those pathways or establish formal convergence.

An adaptive schedule would react to loss or alignment and introduce additional
thresholds, history dependence, and feedback. I recommend fixed schedules for
this first comparison. An adaptive controller becomes a focused follow-up once
the descriptive response and a separate optimization objective are established.

## Preparation before execution

Verify the continuation loader on a small mechanics fixture, including model,
optimizer/scaler/RNG restoration, common g357, unchanged Adam clock, exact data
coverage and LR indexing. Pin the new scientific inputs before executing them.
Retain the parent recovery state, new recovery checkpoints at534 and712,
per-update logs, source snapshots, manifests and progress records. Record global
update, data position and branch identity in every recovery artifact.

Before any billable run, use the Runpod router and live catalog, calibrate this
continuation workload, and state a total duration/cost cap including validation,
storage, retries, transfer and verification. Monitor read-only at bounded
intervals. Retrieve and hash-check all required artifacts before deleting only
task-owned workers/volumes. This recommendation supplies no new spending envelope.

## Authorized first test

The user selected a single seed and minimum elapsed time on2026-10-04.
[Analysis004](../004-2026-10-04-lr-schedule-intervention/README.md) owns six
parallel seed1235 continuations at14M/70M (Runs011-016), with original, held and
accelerated decay. The proposed extra seeds remain deferred validation.
The original design and earlier observations above remain preserved.
