# Run 257: later quarter

H046 / ongoing Analysis065, intervention I14. From untreated state 448,
first-linear weights and biases use Adam LR0.001; later layers use LR0.00025.
All layers remain trainable. There is no inherited freeze and no change at 500.
First affected update448->449; continue through 801 to measure outgoing changes800.
All data/model/optimizer/precision settings and full fork match unchanged Adam.
No reset; native Adam moments and clocks continue with per-layer learning rates.

Primary objective: improve full-window 448-800 mean loss while reducing large
excursions. Also report450-500 event, mean last 32 states 769-800, endpoint800,
positive variation/max upward change, and first10 consecutive states below 0.02
(censored if not reached). Threshold fixed before these runs, not a claim of
optimality. Every arm compares with unchanged baseline, with global/later-layer
controls supplementary. Same multipliers do not match total update norms.

Qualification: untreated448-512 replay and full state 512 equality; observer
preserves RNG/gradients; first changed step has native moments and scaled applied
increments. All 5000 rows retained; dense447-501 gradients/five representations,
rolling cosine/RMS, actual per-layer increments, first moments and denominator
RMS through 800; fork, qualification, first-step, recovery and end checkpoints.
No new SVD or validation result is planned; no architecture change.

Owned Spark, no cloud charge. Prior similar runs8-15s each; allow a few minutes
for six workers plus retrieval and verification.600s per worker limit with630s
detached timeout and30s grace. Six-run maximum63min of compute, under18GB total
retained including archives (under3GB/run). Progress every64 updates, signal/
STOP/time/nonfinite guards; verify sizes/SHA256 after retrieval and confirm exit.
Single seed and fork; no global stability or generalization inference.

## Completed and verified

Mean loss 448-800: **0.01947368**, -14.62%
versus unchanged Adam. Maximum 0.05827207 at 559; loss at 800
0.01692818; last 32 mean 0.01346481. All layers remain
trainable. See [comparison and limitations](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/005-selective-layer-lr.md).

All 67 input identities and 22 retrieved files verified. Exact native replay of
65 states and full state 512, full fork 448 equality, native first-step moments,
scaled increments and independent dense diagnostics pass. Worker elapsed
7.27s; peak allocated GPU memory 400.8 MB.
Worker exited; owned Spark, no cloud charge. Retained raw data and archives have
verified local copies. See verification.json, retrieval.json and the analysis
compute closeout. Scientific configs and executed inputs remain preserved.
