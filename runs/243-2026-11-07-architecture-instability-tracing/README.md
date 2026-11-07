# Run 243: architecture-wide representation tracing

Status: completed, retrieved, verified, cloud workers deleted. Analysis 063 / H043.
The user clarified the targets as ResNet-32/CIFAR-10 and Transformer/WikiText-2,
and explicitly authorized Spark and Runpod for accurate, faster completion.
No architecture, input-normalization or optimizer intervention was performed.

## Scientific contract and result

The frozen config selects recovered loss excursions after retained Run 223
checkpoints 256 and 544, with diagnostic outgoing-update rows 280--310 and 552--600.
Capture states 279--311 and 551--601. Full training coverage remains 50,000 images
or 2,088,600 supervised tokens per update. Representation probes use the original
first 1,000-image BatchNorm group and first 64 language sequences (2,240 supervised
token positions). Model and Adam state come from authenticated seed-0 parent snapshots;
LR 0.003, betas (0.9, 0.999), no decay or clipping, FP32 native, FP64 reductions,
TF32 disabled. Original data ordering and physical groups are preserved.

The variable compared is architectural measurement location, on unchanged
single-seed continuations; objectives, data, optimizer policy and source states
are fixed within each case. The two architectures have different objectives,
probe sizes and time-window lengths, so this is not a matched causal comparison
of architectures or intrinsic rank. No validation/generalization claim is made.

ResNet reproduces all 56 historical loss/norm pairs exactly. Transformer differs
in all 58 recorded pairs, with maximum relative loss discrepancy 0.270%; it is a
qualified continuation from the same saved state, not bitwise historical replay.
The frozen worker's historical-description string mentions Spark; its actual
hardware manifest identifies RTX 4090, and the analysis explicitly corrects this
label while preserving the original string. Model parameters, optimizer state,
source inputs and update policy were not changed by the infrastructure retry.

Results and the didactic algorithm are in
[Analysis 063](../../analyses/063-2026-11-07-architecture-instability-tracing/README.md).
ResNet's early reversals become more closely coupled to output deeper in the
residual stages. Transformer output reverses strongly while the embedding still
moves forward; the first measured transition is across attention. Neither trace
proves a unique initiating cause. MLP causal interventions remain TODO.

## Execution history and resources

1. Owned Spark mechanics calibration initially failed to import threadpoolctl.
   Installed pure-Python threadpoolctl 3.6.0 without upgrading other dependencies;
   both retries completed. These calibrations overwrite one allocated snapshot
   slot at actual states 256--258 / 544--546 and never enter scientific reductions.
   Their manifest capture_states reflects allocated shape; boundaries record the
   actual states. Both workers exited. Measured totals were 201.65 s / 254.06 s.
2. First Runpod RTX 5090 attempt at live USD 1.19/hour failed during environment
   setup before training because the system Python is externally managed. It was
   deleted and its logs/lease preserved. A second 5090 request failed capacity
   without allocating a worker.
3. One RTX 4090 at live USD 0.89/hour used a separate venv with pinned NumPy 2.5.0,
   SciPy 1.18.1 and threadpoolctl 3.6.0, native PyTorch 2.11.0+cu130, and the pinned
   image digest in cloud_lease.json. It had a 30 GB container and 250 GB ephemeral
   volume, USD 5 total ceiling and 2.5-hour maximum lease. Initial qualified ETC
   was 20--45 minutes; workers and retrieval stayed within the approved envelope.
4. ResNet capture completed in 748.6 s and its CPU reduction continued. The unused
   GPU then ran Transformer capture while that reduction finished. Only the
   sequential launcher shell was suspended to prevent a duplicate launch; neither
   scientific worker was suspended or changed. The detached overlap manager
   records both successful completions in cloud_orchestration.json. Full worker
   durations including reduction were 1494.88 s / 753.83 s. Peak CUDA allocations
   were 3.526 GB / 1.746 GB. Read-only monitoring used bounded intervals.
5. All retained files were downloaded and their sizes/SHA-256 checked before the
   task-owned pod was deleted. No unrelated resources were modified. The Spark
   closeout confirms no active task workers. Cloud closeout inventory confirms
   neither task-owned pod remains; no cloud storage is retained.

Provisional wall-clock compute estimate across both allocated attempts:
**USD 0.7306**, excluding unreconciled ephemeral-storage charges and final
billing adjustments. There are no ongoing task-owned cloud charges after deletion.
Owned Spark calibration storage is 6.547 GB
allocated (197.557 GB apparent sparse sizes).
Local retained artifacts including the archive occupy
7.092 GB; ignored staging/scratch occupies
1.039 GB. Spark has no external cloud charge.

## Evidence and verification

- `112` downloaded files, archive
  `3407519465` bytes, SHA-256 `9ae87675fe4741a2aef66f2ee1d0f653650cda8388af21c71381a0cf7320c448`.
- `input_manifest.json`: 44 frozen scientific input identities;
  source/cache/model/data/parent checkpoint identities are pinned and rechecked.
- `artifact_inventory.json`: every raw/retained file's size, hash and retention
  decision. Large dense activation and gradient arrays are derived caches;
  all diagnostic-state model snapshots, end/recovery optimizer/RNG state,
  fixed inputs/targets, full-objective scalars, exact Grams and levels are retained.
  Recomputing raw gradients requires the pinned full objective, not only the probe.
- `verification.json`: full transfer/source checks, complete 33/51 model-snapshot
  sequences, exact final snapshot versus recovery-state model equality, PDF text
  bounds, execution/resource accounting and reduction checks.
- Observer checks preserve gradients, RNG and buffers and match plain forwards;
  all fixed input matrices match exactly. Gram diagonal/adjacent contractions
  are independently checked. Local SVD checks cover PSD, trace/reconstruction,
  and raw-gradient boundary norms/cosines. The Transformer positional/embedding
  Gram scaling agrees with factor 200 to relative error 1.2e-7.
- Correct-label log-probability negative means reproduce probe losses to maximum
  errors 2.95e-8 / 1.16e-7. The common method reproduces all 1,765 prior MLP
  layer cosines exactly and validates all five complete spectra.
- Both two-page PDFs were rendered and visually reviewed. Full figure/source
  hashes, numerical audits and the visual review are recorded in Analysis 063.

## Run-local code

- `diagnostics.py`: snapshots at native boundaries and exact FP64 block Grams.
- `01_worker.py`: unchanged full-objective continuation, probes and checkpoints.
- `02_remote.py`: Spark staging, detached launch and read-only monitoring.
- `03_cloud.py`: bounded cloud lease, transfer verification and cleanup.
- `04_overlap_reduction.py`: unchanged Transformer worker overlapping CPU reduction.
- `05_verify_closeout.py`: source/artifact/snapshot/PDF closeout verification.
- `test_diagnostics.py`: observer, in-place, raw-gradient, Gram, SVD and zero tests.

Scientific executed inputs are frozen. Infrastructure failures retain their
original attempt records; the large caches are reproducible from retained states
under the recorded hardware/runtime contract. Future hardware changes require
fresh numerical qualification. No randomized sketches or per-sample averaging
are used. No manuscript edits or remote publication are part of this run.
