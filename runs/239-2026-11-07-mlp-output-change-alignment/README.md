# Run239: consecutive output-change alignment through state800

H042 / Analysis062 extension explicitly requested for the final row of the
existing figure. Replay the unchanged control0..800 and one additional Adam
update to801. Capture raw FP32 outputs for all5000 fixed examples, all10 output
coordinates, at every state447..801. Define F_t as their concatenation and
B_t=cos(F_(t+1)-F_t,F_t-F_(t-1)), for t448..800. Differences and dot/norm reductions
use FP64. Zero displacement norm gives null. This is global vector alignment,
not average sample-wise cosine, output/target alignment, or output-vector cosine.

All801 retained loss/gradient-norm pairs and800 adjacent gradient statistics
through800 must match Run223. Full model/optimizer/RNG/current/previous gradient
states512/800 must match. State801 is a declared one-update continuation beyond
the reference, needed solely for the last outgoing displacement. Observer
preserves native gradients and RNG. Existing gradient figure rows are reused.

Same seed0, pinned5000-image CIFAR10 training data, 3072-200-200-10 tanh MLP,
all656810 parameters, physical1000, half-squared-error loss, Adam(.9,.999),
LR.001, eps1e-8, no clipping/decay. FP32 training, FP64 diagnostic reductions,
TF32 disabled. Pin all source/cache/reference identities. Persist outputs,
labels, scalar boundaries, final/recovery model/optimizer/RNG/gradient histories.
No new validation, training policy or manuscript changes. Single seed/window.

Owned Spark, zero cloud cost. Expected under3 minutes based on preceding10s
replays; new observer cost unmeasured. Worker300s cap, detached330s timeout with
30s kill grace, progress/recovery every64. Expected raw output plus archive
below250MB. Verify download size/SHA256 and task process exit; preserve all
unrelated workloads and previously executed scientific sources.

## Completion

Completed801 updates in12.297s; all801 retained reference boundaries match,
with full512/800 state equality. Captured355 output arrays and353 global
output-change cosines. Peak GPU allocation241,174,016bytes. All43 source and10
artifact hashes verified. Retained94,962,736 raw bytes plus87,882,363-byte archive;
worker exited, zero cloud cost. See `completion.json` for remote storage.

Analysis062 independently verifies every output cosine, every output-derived
loss, the88-state residual overlap, and the extra800->801 update on CPU. Updated
the existing PDF with final panel(g), retaining all2,118 prior curve values.
The seven-row figure now contains2,471 plotted values and passed visual review.
Executed scientific sources remain unchanged; no manuscript edits.
