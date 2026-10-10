# Run 004: Pythia-70M scale baseline

Status: completed (attempt01), 712 updates, retrieved/verified, cloud closed.
User explicitly authorized both full passes and Runpod launches.

Median adjacent cosine: raw −0.160566, adaptive 0.931795. Final B/A:
raw 0.309380, adaptive 0.609436. Full validation loss ends at 4.063201.
All 55 files (10,707,925,449 bytes) passed strict size/SHA verification and
independent local recovery-state reconstruction. Initial checkpoint hash matches
the frozen Linux seed/recipe identity. Owned Pod and 40GB volume deleted;
cost is provisional. See [the observation](observations/001-gradient-geometry.md)
and [closeout receipt](prelaunch/closeout.json).
Hypothesis H002: adjacent alignment and cumulative geometry vary predictably with model scale.

This run varies the pinned Pythia architecture. It matches Run 002's random
SmallInit/WangInit recipe, seed 1234, token cache and permutation, 712 AdamW
updates, batch 1024 (microbatch 4), optimizer, LR, precision, and measurement
boundaries. Dimensions change, so initial tensors cannot be identical across sizes.
No pretrained weights are loaded. Tokenizer/cache remain the 14M baseline's bytes.

For each raw gradient and adaptive direction, record adjacent cosine,
A_t=sum ||x_i||, B_t=||sum x_i||, B_t/A_t, and ||x_t||. The direction is
bias-corrected m/sqrt(v), with epsilon, before LR/weight decay; its moments use
the clipped gradient. Both streams use identical parameter ordering. Record
the adaptive/raw norm ratio. This is an optimizer-direction statistic, not
an actual parameter-displacement or causal optimization-benefit measurement.

All training and validation tokens, including short tails, are measured. Main
plots omit the final partial batch (update 712), as previously requested; full
data and terminal statistics retain it. Initial/midpoint/final recovery state
includes both streaming accumulators, optimizer, scaler, RNG, and data position.

Model: `EleutherAI/pythia-70m-deduped` at `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
Actual parameters: 70,426,624. Scientific inputs are frozen before
full-pass execution and retained in the attempt source inventory.

One seed per size is a descriptive comparison. Trends can contradict a proposed
monotone expectation, but three sizes cannot establish a predictive scaling law.
Absolute norms and A/B depend on parameter dimension; compare dimensionless
cosines/ratios alongside requested absolute values. Cross-run analysis belongs
to Analysis 001. No new manuscript claim is implied by this run alone.

Execution: one RTX 4090, live $0.74/hour, at most six hours including setup,
calibration, full pass and retrieval. Cohort maximum $10 including storage;
60GB container and 40GB persistent volume per worker, removed after size/SHA
acceptance. Thirty-minute retrieval margin; detached process timeout plus local
deadline guard. Guard needs the controller awake/networked. Monitor bounded
read-only snapshots every 45–60 seconds; stale events, nonfinite metrics,
OOM, overflow retries, or deadline fit are warning conditions.
