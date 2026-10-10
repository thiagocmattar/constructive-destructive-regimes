# Run 015 Pythia 70M hold LR continuation

Status: completed; 356 continuation updates; retrieved and independently verified; worker/volume removed. H004 first intervention, seed1235. Parent is Run007
attempt01 at update356. Continue identical remaining batches through update712.
The user chose a single seed for the first test and fastest parallel execution;
multiple seeds are deferred to validation. Architecture/weights, moments and Adam
clock, scaler, RNG, raw/adaptive diagnostic states and data position are restored.
The future LR rule alone varies. See config.json for the precise intervention.

Raw accumulated unscaled gradients before clipping, all unique trainable
parameters; FP32 vectors and FP64 cosine/norm/cumulative reductions. Existing
adaptive diagnostics remain secondary and exclude LR/decay. Cumulative geometry
inherits the parent prefix; Adam clocks are never reset. Cosine357 is common
pre-intervention; the primary descriptive interval is358-712. No raw history is
discarded, and final partial update712 remains in terminal reductions.

Complete cached MiniPile coverage over parent plus continuation, no repetition;
full token-weighted validation at356/445/534/623/712, including short tails.
Retain new recovery states at534/712 and immutable parent locally. Global step,
order hash, optimizer/scaler/RNG and diagnostic state persist in recovery files.

Six parallel Secure RTX5090 workers, live price checked before creation, at most
two hours per worker and USD15 aggregate including storage and bounded retries.
60GB container/40GB persistent Pod storage, no network volume. Detached execution,
20-minute retrieval margin, local deadline stop guard and bounded read-only
monitoring. Hash-verify downloaded inventory before worker deletion. Unrelated
resources are preserved. No cloud storage is intentionally retained afterward.

Analysis004 owns the comparison. This is a descriptive single-seed intervention,
conditional on a common trained prefix. LR also scales AdamW weight decay;
alignment changes do not establish convergence or a causal performance mechanism.

Verified result and figure interpretation: [Analysis004 observation001](../../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md).
[Retrieval acceptance](prelaunch/attempt1-retrieval.json) and [cloud closeout](prelaunch/closeout.json) preserve source/parent identities, terminal CPU verification, deletion and provisional cost.
