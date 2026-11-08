# Run 278: global functional-step budgets

H048 / Analysis 065. Control the actual complete-network movement measured by the fixed probe, rather than only local affine parameter contributions. Calibrate each tracked representation's coordinate RMS from eight native outgoing updates. After a native Adam proposal, evaluate the candidate, use a global factor at most 1, and backtrack until the output or all tracked representation budgets hold. All moments/clocks and the full training objective remain native; q50 is the primary candidate, q75 exploratory. This bounds probe movement and does not by itself guarantee objective decrease. All layers remain trainable.

## Verified execution and results

Completed 4 matched continuations on owned Spark, with loss states 448-1152 and 2,820 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0174389676. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `output_budget_q50` | +58.18% | +40.70% | +151.23% | +494.28% |
| `representation_budget_q50` | +38.67% | -5.52% | +224.35% | +428.93% |
| `output_budget_q75` | +60.23% | +13.38% | +249.48% | +474.19% |

The independent reduction checks 12 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 48.06 s, with peak allocated CUDA memory 553.1 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run278.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
