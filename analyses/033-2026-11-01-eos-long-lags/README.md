# Analysis033: long-lag full-gradient profiles

Prepared follow-up to [Analysis032](../032-2026-11-01-eos-exact-lag10/REPORT.md)
and the nine [Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md)
conditions. [Run206](../../runs/206-2026-11-01-eos-long-lags/README.md) owns
the scientific inputs and new cloud continuations. Execution has not started.

The requested comparison is the same 27 anchored profiles, at model states
500, 2000 and 3000, with much longer forward lags. We reuse Run205's retained
model/optimizer/RNG states at anchor+10 and its independently verified lags0..10.
No replay from initialization is needed. The two historically unqualified
Run062 reconstructions remain flagged and excluded from primary figures;
25 profiles remain primary. Continued cloud forks are separate trajectories,
not historical states beyond the saved anchor.

All27 profiles reach lag256. Positive-lag1 controls stop at256. For profiles
with a negative original lag1, the horizon doubles through512,1024,2048 until
the median half absolute odd/even cosine gap over the last32 adjacent pairs is
at most0.1. This is a declared descriptive stopping rule, not a stability test.
The final figure will show odd/even branches, common coverage, unequal extended
coverage and every stop reason. Persistent oscillation is a possible result;
the protocol does not require the expected damping to occur.

Each cosine uses the whole raw training-objective gradient, all trainable
parameters, at the optimizer boundary before applying that gradient. Parameters
and accumulated gradients are FP32; diagnostic dot products/norms are FP64,
without projection, centering or smoothing. Every new lag has both CUDA FP64
and independently chunked NumPy FP64 reductions. Zero denominators are undefined.
Only the fixed reference and current/recent gradient are needed in memory.
Full reference histories remain in Run205; the new run retains full gradients
at lag11 and the final odd/even pair, per-lag exact scalar products/norms, and
model/optimizer/RNG recovery state every128 updates and at each terminal state.

Models, seed0, data/order, objectives, LRs and plain SGD match Analysis032.
MLP uses all first5000 CIFAR examples; ResNet uses all50000 with the original
50 fixed1000-image train-mode BN groups; Transformer uses all59680 sequences,
2088600 supervised tokens and the same token-weighted objective. No held-out
evaluation, augmentation, clipping, momentum, decay, schedule or dropout is added.
One Transformer compute chunk is selected from64/128/256/512 using two full
gradient timings each on the late highest-LR state. A candidate must agree with
the original chunk64 to1e-5 relative gradient/1e-5 absolute loss and use<80%VRAM.
The selected chunk is rechecked at all nine Transformer starts. This preserves
the mathematical objective but changes FP32 accumulation partitioning; it need
not reproduce a bitwise-identical long trajectory. ResNet BN groups never change.

The proposed deployment uses nine independent Transformer workers and three
ResNet workers; the cheap MLP windows run first on one ResNet worker. Secure
RTX5090 live price is USD1.19/GPU-hour, with Secure RTX4090 at USD0.89 as the
capacity fallback (2026-11-01 read). The qualified planning range is roughly
1-3 hours; the previous unchanged-chunk4090 measurements were8.8-9.7s per
Transformer full gradient and2.6-3.9s for ResNet. Larger-chunk speedup is
unmeasured until the paid, bounded calibration. The maximum is USD30 aggregate
and four hours, including setup/retrieval margin; a USD25 compute trigger asks
workers to checkpoint and finish, and an independent local guard stops compute
at USD29 or the deadline. Each Pod has30GB container disk and8GB Pod volume;
no network volume is created. Detached jobs persist progress and logs, with
approximately one-minute reads. Each worker is deleted only after local file
size/SHA-256 verification; task storage is then removed. No unrelated resource
or Spark workload is modified. The larger spending envelope awaits approval.

`00_prepare.py` verifies source inventories and builds per-worker packets;
`01_cloud.py` handles task-owned provisioning, calibration, detached execution,
monitoring, verified retrieval and cleanup. `02_analyze.py` will audit the retained
vectors and scalar reductions, preserve every measured lag and make PDF figures.
There is one seed and one dependent anchored pair per lag. The adaptive horizons
do not supply sampling uncertainty, stationarity, generalization or loss-benefit
evidence.
