# Run 285: ResNet probe-loss backtracking

H048 / Analysis 065. Transfer the MLP loss-aware comparator without threshold tuning. Backtrack the full native Adam increment using loss on the intact 1000-example probe normalization group; retain the full 50000-example training objective. Reuse Run 270's matched same-worker control after exact fork/two-step and config/input checks, preserving raw bytes and provenance. No guarantee of full-objective monotonicity outside the probe.

## Verified execution and results

Completed 2 matched arm records over loss states 264-576, with 1 newly executed continuations and 313 newly applied updates including terminal outgoing diagnostics. Every intervention uses unchanged Adam on the same worker as its baseline; the full 50,000-example objective is preserved.

The control is an authenticated reuse of Run 270 on this same worker, with exact full-fork/native-continuation checks and byte-identical artifacts. It is excluded from new-update and timing totals.

Percent changes below use the complete loss window. Tail is the last 32-state mean; PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `probe_loss_guard` | +36.46% | +1486.43% | -13.82% | -97.17% |

Independent reduction checks 6 retained arm-state audits, source and retrieval identities, raw gradients, gate recurrence/placement, actual increments, optimizer clocks and exact branch identity. Original-hardware triplet replay verifies representation RMS and cosines. Independent replay checks before/accepted probe losses and all recorded dyadic acceptance decisions. The worker was deleted after verified retrieval.

New continuation durations sum to 1,143.90 s; peak allocated CUDA memory is 5,330.5 MB. Durations include tracing, audit writes and terminal evaluation, but exclude prefix, mechanics, replay, setup and transfers. They are against instrumented controls, not uninstrumented Adam. This run shares a billed worker with other H048 runs; task-wide cost/storage accounting is in the [compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Terminal test loss is secondary and uses all 10,000 test examples at state 576:

| Arm | Test cross-entropy |
|---|---:|
| `control` | 1.84433407 |
| `probe_loss_guard` | 1.23735155 |

[Verified numerical comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run285.json) and [methods, interpretation and figures](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored `artifacts/`; scientific inputs remain frozen.
