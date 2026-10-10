# Run 021: Pythia-14M warmup-only LR 0.001

Status: completed all 712 updates; full verification, retrieval and PDF visual
review passed. Finished 2026-10-09 02:58:54 UTC. User explicitly authorized
full-pass execution on the owned DGX Spark. Condition: base.

This run shares Run002's exact frozen initial tensors and seed 1234 data order
with Runs 021–023. One complete MiniPile pass: 712 updates, effective batch 1024,
microbatch 4, sequence length 2048, FP16 compute with FP32 AdamW parameters and
moments. The existing pre-step 1% warmup starts at LR = 0; from update 9,
LR = 0.001 for the remainder of the pass.

Every accepted update captures individual norm, A, B, B/A and adjacent cosine
for raw gradient, adaptive direction and stored momentum m, with paired norm
ratios. Momentum is actual uncorrected FP32 Adam `exp_avg`, whose inputs are
clipped gradients; prefix accumulation and reductions are FP64. Both complete
validation passes and all token tails are included.

Recovery states are retained initially, at midpoint, at the final update and
every 100 updates, including all three streaming buffers, optimizer, scaler,
RNG and data position. The detached sequential queue limits each condition to
8 hours. Calibration is separate and reset; three calibration updates passed
independent verification before launch.

Scientific bytes are frozen in the shared source inventory/packet. See the
[full design, metric contract and monitoring instructions](../../analyses/008-2026-10-08-no-decay-momentum/README.md).
Local retrieval verified all 48 output files (7.206 GB) by size and SHA-256.
The [raw/adaptive comparison](figures/raw_adaptive_comparison.pdf) and
[momentum figure](figures/momentum_evolution.pdf) were rendered and visually
reviewed. Numerical reductions and caveats are in
[Observation001](observations/001-three-stream-geometry.md). The
[final joint comparison](../../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/README.md)
includes all three completed, verified runs. Remote state is preserved.

No stability/generalization claim or independent-seed uncertainty is implied.
