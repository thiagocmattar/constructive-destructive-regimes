# Run 270: resnet representation-informed interventions

H048 / ongoing Analysis 065. Planned before execution; see config.json. All arms
compare to an unchanged same-worker control. Fixed native settings and full
objectives are preserved. Thresholds use the eight untreated updates immediately
before the fork; no treatment outcomes set them. Single seed and selected fork;
this tests transfer across these architectures, not universal generalization.

All first-layer/entry caps modify post-Adam increments. Native moments and clocks
advance. The functional rule caps local parameter-induced preactivation change
at the current fixed probe input, not the full coupled downstream displacement.
ResNet uses an intact 1000-image BN group. Transformer uses 64 sequences and
2240 supervised tokens; training still uses every supervised training token.
MLP uses all 5000 training examples. Other representation diagnostics preserve
individual sample/coordinate structure, with no sample averaging before cosine.

Scientific inputs are frozen; changed policies receive a new run ID.

## Verified execution and results

Completed 7 matched arm records over loss states 264-576, with 7 newly executed continuations and 2,191 newly applied updates including terminal outgoing diagnostics. Every intervention uses unchanged Adam on the same worker as its baseline; the full 50,000-example objective is preserved.

Percent changes below use the complete loss window. Tail is the last 32-state mean; PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `discard_entry` | +124.20% | +1996.71% | +352.43% | +124.15% |
| `cap_entry` | +54.59% | +332.16% | +152.64% | +67.91% |
| `functional_entry` | -6.29% | -22.76% | +21.44% | -17.15% |
| `block_shock` | +255.72% | +4335.72% | +433.26% | +191.40% |
| `global_half` | +17.05% | +406.57% | +64.02% | +7.53% |
| `functional_trunk` | -17.42% | -8.31% | -13.82% | -17.71% |

Independent reduction checks 21 retained arm-state audits, source and retrieval identities, raw gradients, gate recurrence/placement, actual increments, optimizer clocks and exact branch identity. Original-hardware triplet replay verifies representation RMS and cosines. The worker was deleted after verified retrieval.

New continuation durations sum to 7,860.18 s; peak allocated CUDA memory is 5,346.5 MB. Durations include tracing, audit writes and terminal evaluation, but exclude prefix, mechanics, replay, setup and transfers. They are against instrumented controls, not uninstrumented Adam. This run shares a billed worker with other H048 runs; task-wide cost/storage accounting is in the [compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Terminal test loss is secondary and uses all 10,000 test examples at state 576:

| Arm | Test cross-entropy |
|---|---:|
| `control` | 1.84433407 |
| `discard_entry` | 1.14983099 |
| `cap_entry` | 1.54713519 |
| `functional_entry` | 1.93031223 |
| `block_shock` | 0.83692602 |
| `global_half` | 1.57435569 |
| `functional_trunk` | 1.97763503 |

[Verified numerical comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_resnet.json) and [methods, interpretation and figures](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored `artifacts/`; scientific inputs remain frozen.
