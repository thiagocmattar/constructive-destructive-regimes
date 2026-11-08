# Run 274: transformer distributed displacement cap

H048 / Analysis 065. Transfer the prefix-q50 local affine cap throughout the designated representation-producing trunk. Selected from MLP exploration; no target-architecture treatment outcomes available at selection. All remaining matched settings, full objectives and audit retention follow Run 271. See config and input manifest. Same-worker untreated control required.

## Verified execution and results

Completed three matched arm records over loss states 552-928, with 3 newly executed continuations and 1,131 newly applied updates including terminal outgoing diagnostics. The full 2,088,600-token training objective is preserved; the fixed probe contains 2,240 supervised positions from 64 sequences.

The first cloud attempt completed mechanics only and was cancelled before scientific training. The replacement RTX 5090 worker completed the unchanged scientific design; the original attempt and its cost remain recorded. Both interventions use this worker's own unchanged Adam control.

Percent changes below use the complete loss window. Tail is the last 32-state mean; PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `functional_trunk_q50` | -26.00% | -73.36% | -79.63% | -98.81% |
| `embedding_x4_functional_trunk_q50` | -9.38% | -66.58% | -61.34% | -88.04% |

Independent reduction checks nine retained arm-state audits, source and retrieval identities, raw gradients, gate recurrences/placement, actual increments, clocks and branch identity. Original-hardware triplet replay verifies representation RMS and cosines. The worker was deleted only after verified retrieval.

New continuation durations sum to 8,949.45 s; peak allocated CUDA memory is 4,044.8 MB. Timers include tracing, audit writes and terminal evaluation; they exclude prefix, mechanics, independent replay, setup and transfers. They compare instrumented policies/controls, not uninstrumented Adam. Task-wide cost/storage accounting is in the [compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Terminal losses retain the complete cached splits with token weighting. Exact valid-token coverage is shown below; no truncation is used.

| Arm | Split | NLL | Supervised tokens |
|---|---|---:|---:|
| `control` | valid | 10.15664667 | 217,630 |
| `control` | test | 9.97783304 | 245,550 |
| `functional_trunk_q50` | valid | 20.20185774 | 217,630 |
| `functional_trunk_q50` | test | 19.97274356 | 245,550 |
| `embedding_x4_functional_trunk_q50` | valid | 18.16070777 | 217,630 |
| `embedding_x4_functional_trunk_q50` | test | 17.89301201 | 245,550 |

Terminal comparisons do not establish held-out trajectory rankings. RTX 4090 historical controls follow a different numerical trajectory and are not substituted for this worker's control.

[Verified numerical comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run274.json) and [methods, interpretation and figures](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored `artifacts/`; scientific inputs remain frozen.
