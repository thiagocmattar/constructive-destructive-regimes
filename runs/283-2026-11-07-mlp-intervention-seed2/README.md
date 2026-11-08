# Run 283: MLP seed 2 intervention replication

H048 / Analysis 065. Reuse the pinned existing seed 2 initialization with the same data, objective, native Adam settings and 448-1152 window. Recalibrate the same rules from the eight preceding untreated updates. Compare distributed neuron q50, output q50 with 1% factor recovery, uniform half increments, and unchanged Adam. No seed-specific selection or tuning.

## Verified execution and results

Completed 4 matched continuations on owned Spark, with loss states 448-1152 and 2,820 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0181462186. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `functional_trunk_q50` | -32.08% | -64.49% | -51.62% | -82.56% |
| `output_budget_slow` | -24.10% | -54.30% | -50.55% | -80.35% |
| `global_half` | -17.08% | -62.32% | +18.21% | -60.29% |

The independent reduction checks 12 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 279.72 s, with peak allocated CUDA memory 591.7 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run283.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
