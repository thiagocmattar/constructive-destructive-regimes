# Run 271: transformer representation-informed interventions

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

Completed eight arm records on two RTX 4090 workers (seven unique arm names, with one independently executed unchanged control per worker). Each spans loss states 552-928 and applies 377 updates including the final outgoing diagnostic: 3,016 new updates total. Both controls have exactly identical full-window loss histories; all effects below use the corresponding worker control.

| Worker | Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---|---:|---:|---:|---:|
| transformer-a | `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| transformer-a | `discard_entry` | -6.22% | -18.02% | +0.04% | -31.11% |
| transformer-a | `cap_entry` | -9.13% | -22.32% | -23.01% | -49.01% |
| transformer-a | `functional_entry` | -8.53% | -24.58% | -14.68% | -45.20% |
| transformer-b | `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| transformer-b | `global_half` | -16.02% | -19.08% | -55.42% | -87.11% |
| transformer-b | `embedding_x2` | -0.20% | -14.77% | -22.13% | -1.53% |
| transformer-b | `embedding_x4` | +11.52% | -16.34% | -9.72% | -0.05% |

Tail is the last 32 loss states; PV is total positive adjacent loss variation. Configured block-shock and embedding-plus-block-shock arms were not executed after the MLP negative results; their Transformer efficacy is unknown. All three requested selective criteria were executed.

All 24 retained arm-state audits pass source/retrieval identity, gradient and increment checks, gate recurrence/placement, clocks and exact fork qualification. Same-hardware representation triplets were independently replayed. Both task workers were deleted after verified retrieval.

Continuation durations sum to 31,937.90 s; peak allocated CUDA memory is 4,026.0 MB. These include tracing, audit writes and terminal evaluation, and exclude prefix/mechanics/replay/transfers/setup. Timing compares instrumented continuations, not plain Adam. Cloud and local storage are reported in the [task compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Complete terminal validation/test NLLs are secondary (217,630 / 245,550 supervised tokens), both at state 928:

| Worker | Arm | Validation NLL | Test NLL |
|---|---|---:|---:|
| transformer-a | `control` | 16.40552034 | 16.20191825 |
| transformer-a | `discard_entry` | 17.27888990 | 17.05959415 |
| transformer-a | `cap_entry` | 17.93532906 | 17.71591817 |
| transformer-a | `functional_entry` | 18.18301882 | 17.95792347 |
| transformer-b | `control` | 16.40552034 | 16.20191825 |
| transformer-b | `global_half` | 17.43668628 | 17.22170812 |
| transformer-b | `embedding_x2` | 17.02201868 | 16.77441840 |
| transformer-b | `embedding_x4` | 16.16831257 | 15.90729007 |

[Verified numerical record](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison.json), [methods and conclusions](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored artifacts; scientific inputs are frozen.
