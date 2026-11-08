# Run 281: Transformer probe-based damping

H048 / Analysis 065. Transfer output-q50 damping with 1% factor recovery and the probe-loss backtracking comparator. Keep the full objective, native Adam and fixed 64-sequence probe. The latter uses labels and bounds only probe loss. Reuse Run 274's same-worker control only after exact full-fork/native-two-step and source/config identity checks, preserving raw bytes and provenance. The prelaunch-only draft is retained in ignored scratch; no scientific execution preceded this design revision.

## Verified execution and results

Completed three matched arm records over loss states 552-928, with 2 newly executed continuations and 754 newly applied updates including terminal outgoing diagnostics. The full 2,088,600-token training objective is preserved; the fixed probe contains 2,240 supervised positions from 64 sequences.

The two new branches reuse Run 274's same-worker control after exact full-fork, scientific identity, native continuation and raw-file checks. That record is not a new replicate or newly executed update.

Percent changes below use the complete loss window. Tail is the last 32-state mean; PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `output_budget_slow` | -20.47% | -68.04% | -79.63% | -99.99% |
| `probe_loss_guard` | -22.51% | -69.08% | -79.63% | -99.83% |

Independent reduction checks nine retained arm-state audits, source and retrieval identities, raw gradients, gate recurrences/placement, actual increments, clocks and branch identity. Original-hardware triplet replay verifies representation RMS and cosines. Probe-loss replay additionally verifies the loss guard and its dyadic acceptance decisions; six of the nine arm-state audits concern new branches. The worker was deleted only after verified retrieval.

New continuation durations sum to 5,932.63 s; peak allocated CUDA memory is 5,028.0 MB. Timers include tracing, audit writes and terminal evaluation; they exclude prefix, mechanics, independent replay, setup and transfers. They compare instrumented policies/controls, not uninstrumented Adam. Task-wide cost/storage accounting is in the [compute closeout](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json).

Terminal losses retain the complete cached splits with token weighting. Exact valid-token coverage is shown below; no truncation is used.

| Arm | Split | NLL | Supervised tokens |
|---|---|---:|---:|
| `control` | valid | 10.15664667 | 217,630 |
| `control` | test | 9.97783304 | 245,550 |
| `output_budget_slow` | valid | 16.11932869 | 217,630 |
| `output_budget_slow` | test | 15.92538046 | 245,550 |
| `probe_loss_guard` | valid | 16.55528857 | 217,630 |
| `probe_loss_guard` | test | 16.35442864 | 245,550 |

Terminal comparisons do not establish held-out trajectory rankings. RTX 4090 historical controls follow a different numerical trajectory and are not substituted for this worker's control.

[Verified numerical comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run281.json) and [methods, interpretation and figures](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md). Dense raw evidence remains in ignored `artifacts/`; scientific inputs remain frozen.
