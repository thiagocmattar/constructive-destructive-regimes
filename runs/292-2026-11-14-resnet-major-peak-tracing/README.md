# Run 292: ResNet temporal structure at the larger loss event

Completed and verified for Analysis 063 / H043. The user requested that the temporal-structure
companion match the loss/motion figure at completed steps **390--450**.

- Replay the same unchanged seed-0 Run 273 control from its authenticated state 264.
  Keep Adam, LR 0.003, full 50,000-example objective, physical 1,000-image groups,
  model mode, data ordering and fixed first 1,000-image probe unchanged.
- Capture six displayed stage endpoints plus raw gradients. Capture states 388--450;
  fit 61 incoming representation displacements at 390--450 and the gradients used
  for those updates at 389--449. The predecessor is excluded from both fits.
- Refit exact FP64 temporal Grams/SVD, all stage energies and output overlap, and
  the signed leading output coefficient. No cropping of the previous SVD basis.
- Require exact replay losses/initial gradient and the retained state 296 full
  parameter/gradient audit; independently match new Gram scalars to Run 273.
- Retain 63 model snapshots, gradients, fixed probe, exact Grams, source identities
  and recovery state; hash large reproducible activation caches before discarding.
- One RTX 4090 with the original pinned PyTorch 2.11/CUDA 13 image. Live price and
  lease recorded separately. Qualified estimate 20--40 minutes including transfer;
  hard total ceiling 2 hours/USD 2, 30 GB container +40 GB Pod-local disk, no network
  volume. Detached execution; progress every 30 seconds; verify transfers and delete.
- A preliminary local RTX 5070 Ti/PyTorch 2.11+cu128 gradient check failed exact replay
  (relative gradient error 8.9631e-5 at 264); it is mechanics only, excluded from evidence.

Reproduce with `02_cloud.py prepare`, then `02_cloud.py launch`. Executed
scientific inputs are frozen once launched. The result is descriptive evidence
for one selected native-control event, not a new intervention or generalization test.

## Verified result and closeout

- All 187 native-control losses at 264--450 match exactly. The initial gradient and
  retained state 296 full parameters/gradient match bitwise. All six displayed stage
  cosine/RMS histories match the existing loss/motion figure; maximum absolute
  cosine discrepancy 1.34e-15. Observer and independent Gram checks pass.
- Fresh 61-row output SVD: top-one 27.26%, top-ten 87.06%, rank95=20, rank99=35.
  First-convolution/final-stage output temporal overlap is .567/.985. Raw-gradient
  top-ten energy 94.74%; its rank95 is 11. The leading output coefficient has 33 sign
  changes across 60 adjacent fitted pairs; small coefficient flips are not alarms.
- [Analysis and current figure](../../analyses/063-2026-11-07-architecture-instability-tracing/observations/008-resnet-major-peak-temporal.md),
  [numerical verification](verification.json), [retrieval](retrieval.json),
  [compute closeout](compute_closeout.json).
- Native RTX 4090, PyTorch 2.11.0+cu130, cuDNN 91900; pinned NumPy 2.5.0, SciPy 1.18.1,
  threadpoolctl 3.6.0. Worker replay/capture/reduction took 685.99 seconds with 3.53 GB peak
  allocated CUDA memory. The 186 updates preserve the original complete objective.
- Direct SCP was unusually slow, including a Spark relay attempt. Enabling an
  authenticated temporary HTTPS transfer restarted the Pod before any scientific
  execution; fresh SSH mapping came from `pod get`. All inputs then passed the
  original archive/file hashes. Scientific inputs and the sole scientific attempt
  were unchanged. The transfer token and relay private key were removed.
- Retrieved 75 files, 266.79 MB, plus 161.96 MB archive. Retain 63 model snapshots, full
  gradients, fixed probe, exact Grams, full boundary scalars and recovery state.
  All 11.75 GB of raw/generated files were inventoried; large activation arrays are
  reproducible derived caches discarded with the Pod. The input archive adds 284.98 MB
  locally. Spark retains only 225 bytes of relay logs/host identity.
- The task Pod and its temporary HTTPS service were deleted after verified
  retrieval; fresh account read confirms absence. No cloud storage remains.
  Estimated compute USD 0.412; provider billing is not yet posted. This is below
  the USD 2/two-hour ceiling. No other resources were changed.
