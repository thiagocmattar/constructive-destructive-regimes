# Run 023: Pythia-14M warmup-only LR 0.01

Status: completed; started 2026-10-09 07:59:22 UTC after Runs021/022 completed
and finished at 12:59:25 UTC. All 712 updates and complete token/validation
coverage passed verification; zero overflow retries. Final validation loss:
6.636489, compared with 4.763734 / 6.232207 for Runs021/022. User explicitly
authorized full-pass execution on the owned DGX Spark. Condition: large.

This run shares Run002's exact frozen initial tensors and seed 1234 data order
with Runs 021–023. One complete MiniPile pass: 712 updates, effective batch 1024,
microbatch 4, sequence length 2048, FP16 compute with FP32 AdamW parameters and
moments. The existing pre-step 1% warmup starts at LR = 0; from update 9,
LR = 0.01 for the remainder of the pass.

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
Local retrieval verified all 48 output files (7.206 GB) by size and SHA-256;
all 48 were independently rechecked at closeout. The
[raw/adaptive comparison](figures/raw_adaptive_comparison.pdf) and
[momentum figure](figures/momentum_evolution.pdf) passed Poppler visual review.
See [Observation001](observations/001-three-stream-geometry.md) and the
[final three-LR comparison](../../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/README.md).
Remote state is preserved; no new experiment or cloud rental was launched.

No stability/generalization claim or independent-seed uncertainty is implied.
