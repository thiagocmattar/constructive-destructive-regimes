# Run 054: Pythia-14M, constant post-warmup LR 0.04

Status: launched in the detached Spark queue; user authorized training on the owned DGX Spark while away.
H014; matched frozen initialization, seed1234 order and complete MiniPile pass
with Spark Run023. Only peak LR changes; AdamW weight decay0.1 remains.
712 accepted updates; batch1024/microbatch4/sequence2048, FP16 compute and
FP32 parameters/moments. Capture raw, stored momentum and adaptive norm,
adjacent cosine, A, B and B/A at every accepted update with FP64 reductions.
Raw is pre-clipping; momentum is actual uncorrected exp_avg from clipped g.
Full initial/final validation and partial token tails are retained.
Initial/midpoint/final/every100-update recovery; separate reset calibration.
8-hour full-pass bound; failures preserve evidence and do not lower LR.
[Design and automatic completion](../../analyses/015-2026-10-10-spark-high-lr/README.md).
No general stability threshold or independent-seed uncertainty is asserted.
