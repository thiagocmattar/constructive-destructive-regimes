# Run 002: Adam adaptive direction versus raw-gradient geometry

Status: completed; all artifacts retrieved and verified; both owned Pods removed.
Successful worker (deleted): `9tw9vb6d8m4mg4`, Secure RTX 4090, EU-RO-1, USD0.74/hour.
Lease started 2026-10-03 21:11:04 UTC; controller deadline 2026-10-04 00:11:04 UTC.
Scientific execution: 2026-10-03 22:01:04–23:22:30 UTC, 81 minutes 26 seconds.
Cloud cleanup completed 23:28:33 UTC. Estimated aggregate GPU/disk cost is USD1.71,
including the retired calibration attempt; provider billing remains provisional.
No task-owned cloud storage remains. See `prelaunch/closeout.json` and transfer receipts.

Attempt 01 was retired after three mechanics calibrations measured about 30 seconds
per update. Its 56 artifacts were retrieved and individually verified before Pod
deletion. Attempt 02 preserves the same scientific inputs and uses local disk for
the verified token cache. The aggregate deadline includes both attempts.
Three-step calibration measured about 6.8 seconds per update; projected training
time is about 81 minutes, plus validation/checkpoint/retrieval allowance. No
overflow retries occurred in calibration. Peak allocated CUDA memory was 5.85 GB.

## Results and figures

The complete pass covers 712 updates, 728,375 packed sequences, 1,491,711,416 input
tokens and 1,490,983,041 supervised targets. No data was repeated or dropped.
Initial/full final validation loss: 11.032736 → 5.182392 (500 documents, including tail).

| Full-pass metric | Raw gradient | Adam adaptive direction |
| --- | ---: | ---: |
| Median adjacent cosine | 0.007220 | 0.966250 |
| Negative pairs / defined pairs | 350/711 | 0/711 |
| Median individual norm | 0.409031 | 1250.005724 |
| Final A: sum of norms | 319.486529 | 946630.439779 |
| Final B: norm of vector sum | 107.709080 | 667166.451433 |
| Final B/A | 0.337132 | 0.704780 |

Median paired adaptive/raw norm ratio: 3,175.946. These directions exclude LR
and weight decay, so this is not a ratio of actual physical parameter-step norms.
H001's higher median cosine and higher final B/A are observed in this trajectory.
A and B remain separated: adaptive B is 70.48% of A, a 29.52% relative gap.
Adaptive B/A bottoms at 0.544333 (step 102) then rises; raw B/A bottoms at
0.306944 (step 454). Raw middle-quarter alignment is negative; adaptive alignment
stays positive throughout. Neither alignment nor this norm ratio establishes stability.

- [Paired comparison and norm scales](figures/raw_adaptive_comparison.pdf)
- [Adaptive direction evolution](figures/adaptive_evolution.pdf)
- [Raw-gradient evolution in this run](figures/raw_gradient_evolution.pdf)
- [Observation, trajectory quarters, coverage, and interpretation limits](observations/001-adaptive-comparison.md)
- [Exact figure data and source hashes](figures/figure_data.json)

All main plots display updates 1–711; update 712 is preserved in raw evidence and
full-pass numerical results. All three publication PDFs were rendered and inspected.

The initialization tensors are identical to Run001, and data order/scientific
settings match. Replay is not bitwise identical: median absolute per-step raw
cosine difference is 0.095492 (maximum 1.097571), while final raw B/A differs by
0.002837 and validation loss by 0.000571. Every raw/adaptive comparison above
uses paired measurements from Run002. No single cause of the replay differences
is established. See `prelaunch/replay-comparison.json`.

Acceptance: 25 local checks, five fused CUDA checks, remote/local full-pass
mathematical verification, identical executed scientific source/config hashes,
and strict archive/file-size/SHA-256 checks passed. Attempt 02 retains 57 files
(2,141,378,981 bytes), including initial/midpoint/final recovery checkpoints and
both streaming states. Raw norms agree with the independent clipping reduction
to maximum relative error 1.43e-7. Plotting reductions were finalized locally;
the uploaded plotting draft remains in the frozen source archive, and figure
data identifies the actual generating script. Manuscript originals are intact.

The user requests a new full-pass experiment measuring the same geometry on
the adaptive Adam direction, and comparing its norm with the raw-gradient norm.
They explicitly chose the adaptive direction only: exclude learning-rate scaling
and weight decay. The earlier instruction not to rerun is superseded by this
new explicit experiment request. Runpod launches are already authorized.

## Scientific comparison and H001

H001: along the same AdamW trajectory, the bias-corrected adaptive direction has
higher adjacent cosine alignment and a cumulative resultant closer to the sum of
individual norms than the raw gradient. The user's expectation is that A and B
will be close, with strongly positive adjacent cosines. We report both
distributions and trajectories, their paired differences, and final B/A; no
significance threshold or cross-seed generality is claimed from one trajectory.

Matched to Run001: model architecture/revision, random initialization recipe,
model/data-order seed 1234, pinned MiniPile token bytes/order, packing/causal loss,
full pass including its short tail, optimizer/grouping/hyperparameters, scheduled
learning rate, global clipping, FP16 scaling, FP32 parameters/moments, and full
validation. The added diagnostics read state without changing training. Record
initial model identity and compare selected loss/raw-metric events against Run001
to assess replay agreement; do not promise bitwise replay on different hardware.

## Measurement contract

One step is an accepted optimizer update after accumulated token-weighted mean
next-token loss over the actual batch. All unique trainable parameters, in a
fixed common order, enter both vectors. Raw g is unscaled and captured before
clipping. Adam moments use the globally clipped gradient, matching training.
With optimizer moment step t starting at 1:

```text
d_t = [m_t / (1 - beta1^t)] / [sqrt(v_t / (1 - beta2^t)) + epsilon]
rho_t(x) = dot(x_(t-1), x_t) / (norm(x_(t-1)) * norm(x_t))
A_t(x) = sum_(u<=t) norm(x_u)
B_t(x) = norm(sum_(u<=t) x_u)
R_t(x) = B_t(x) / A_t(x), for x in {g, d}
```

Direction d is computed after the optimizer step from its actual updated FP32
moments, with FP64 evaluation of the formula. It excludes LR, weight decay,
the minus sign of descent, and finite-precision parameter-subtraction effects.
It is not an actual parameter displacement. The first scheduled LR is zero;
moments still update and d is measured, with first cosine undefined.

Per update, save raw and adaptive norms, cosines, A/B/R and adaptive/raw norm
ratio. Zero-denominator ratios/cosines are null with explicit reasons. Prefix
sums/reductions are FP64; previous raw vector is FP32 and previous adaptive
vector FP64. Retain both streaming states in initial/midpoint/final recovery
checkpoints, with optimizer, loss scaler, RNG state and data position. Full vector
histories are not retained. Diagnostic persistent memory is 28 bytes per
parameter (about 394 MB); temporary FP64 direction buffers add about 113 MB.

The final update has 311 sequences; updates 1–711 have 1,024 each. All are
measured and preserved. Main comparison plots show the full-sized updates 1–711
as requested for the prior figure; final full-pass results are separately labeled.
Validation covers all 500 validation documents, token weighted, including tail.

Norm scales differ because preconditioning changes coordinates/scaling. Their
ratio is descriptive, not an energy-efficiency or stability statistic. Positive
adjacent d cosine does not ensure A≈B: A/B depends on all temporal lags. This run
does not estimate per-sample gradients, causal benefit, or generalization across
seeds. No manuscript empirical claims are added automatically.

## Execution and acceptance

One Secure RTX 4090 Pod; three-hour maximum lease and USD3 aggregate target,
including storage/setup/retries. Live price, absolute deadline, current calibration,
input/source identities, and launch records will be recorded in `prelaunch/`.
Train detached with persistent events/manifests and both streaming recovery states.
Monitor progress, loss, throughput, refreshed ETC, memory and warning conditions.
Retrieve and verify sizes/SHA-256 and scientific identities before deletion.
No network volume or task-owned cloud storage is retained after closeout.

Shared Run001 primitives are promoted to `src/baseline/`; original executed Run001
sources remain unchanged. Novel Adam diagnostics and orchestration stay run-local.
