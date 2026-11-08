# Run 273: resnet distributed displacement cap

H048 / Analysis 065. Transfer the prefix-q50 local affine cap throughout the designated representation-producing trunk. Selected from MLP exploration; no target-architecture treatment outcomes available at selection. All remaining matched settings, full objectives and audit retention follow Run 270. See config and input manifest. Same-worker untreated control required.

## Verified execution and results

Completed 2 matched arm records over loss states 264-576, with 2 newly executed continuations and 626 newly applied updates including terminal outgoing diagnostics. Every intervention uses unchanged Adam on the same worker as its baseline; the full 50,000-example objective is preserved.

Percent changes below use the complete loss window. Tail is the last 32-state mean; PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `functional_trunk_q50` | -16.22% | -7.48% | -13.82% | -46.78% |

Independent reduction checks 6 retained arm-state audits, source and retrieval identities, raw gradients, gate recurrence/placement, actual increments, optimizer clocks and exact branch identity. Original-hardware triplet replay verifies representation RMS and cosines. The worker was deleted after verified retrieval.

New continuation durations sum to 2,305.09 s; peak allocated CUDA memory is 5,346.4 MB. Durations include tracing, audit writes and terminal evaluation, but exclude prefix, mechanics, replay, setup and transfers. They are against instrumented controls, not uninstrumented Adam. This run shares a billed worker with other H048 runs; task-wide cost/storage accounting is in the [compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Terminal test loss is secondary and uses all 10,000 test examples at state 576:

| Arm | Test cross-entropy |
|---|---:|
| `control` | 1.84433407 |
| `functional_trunk_q50` | 1.99519318 |

[Verified numerical comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run273.json) and [methods, interpretation and figures](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored `artifacts/`; scientific inputs remain frozen.
