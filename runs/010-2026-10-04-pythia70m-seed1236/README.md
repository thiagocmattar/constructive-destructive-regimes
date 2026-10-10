# Run 010: Pythia-70M, seed 1236

Status: completed all 712 updates; artifacts retrieved and independently verified; worker and attached volume deleted.

This H002 replication changes model initialization and data permutation seeds
to 1236. The same seed and exact batch order are paired across 14M/31M/70M;
the order differs between seed blocks. Both sources of training variability
change together; their effects are not separately identified. The reference is
Run 004, seed 1234. No pretrained weights are loaded.

Match pinned model architecture within size, token/cache bytes, full MiniPile
coverage, 712 updates, 1024 sequences per full update, microbatch 4, AdamW/LR,
clipping, loss scaling, validation, and diagnostic placement. Model shapes differ
across sizes; the initialization recipe is shared, not the tensor values.

Measure each update for raw gradients g (unscaled accumulated batch mean before
clipping) and adaptive Adam directions d (bias-corrected actual FP32 moments,
evaluated in FP64 after the optimizer step). d excludes learning rate and weight
decay; moments use clipped gradients. For each stream record adjacent cosine,
A=sum of vector norms, B=norm of cumulative vector sum, B/A, individual norm,
and the paired adaptive/raw norm ratio. Initial cosine is undefined with a reason.
All unique trainable parameters; FP64 cumulative sums/reductions.

Consume all 1,491,711,416 input tokens once, including the short tail. Validate
all 500 documents with token weighting and tail inclusion. Retain every update;
main evolution plots omit final partial update 712 as previously requested.
Preserve initial/midpoint/final models and recovery state (optimizer, scaler,
streaming diagnostic vectors, RNG, data position).

Execution: one RTX 5090 at live $0.99/hour, or RTX 4090 capacity fallback at
$0.74/hour, Secure Cloud. At most four hours per worker, $25 for all six including
storage and bounded retries. 60GB container and 40GB persistent storage per Pod;
no network volume. Detached training, timeout with 30-minute retrieval margin,
local deadline guard, and bounded read-only monitoring. Calibrate actual hardware
before the full pass, retaining environment/hardware and exact input identities.
Retrieve and verify every file size/SHA and scientific artifacts before deletion.
No task cloud storage is intentionally retained. Hardware differs from the
original RTX4090 cohort; numeric kernels may affect trajectories and are recorded.

Analysis002 combines three seeds per size. Compare full-pass medians, terminal
A/B/B/A, norms, negative-cosine fractions, and per-seed trajectories. Report all
seed points, mean/sample SD and range; paired cross-size differences and ordering
per seed. Three seeds are a small reproducibility check, not a scaling-law
validation or proof of causal optimization benefit. Steps are not independent
replicates. Do not infer stability/generalization from alignment alone.

## Completed evidence

[Observation001](observations/001-seed-geometry.md) records raw/adaptive metrics, artifact verification and cleanup.
[Analysis002](../../analyses/002-2026-10-04-scale-seed-replication/README.md) compares all three seeds per size.
