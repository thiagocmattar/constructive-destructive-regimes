# Run 272: MLP displacement confirmation

H048 / Analysis 065. Extend the selected q50 entry cap and uniform half increments through 1152, with held-forward 801-1152 reporting. Explore the same fixed caps on both hidden affine layers and a uniform quarter increment control. Same unchanged Adam baseline, native training objective, eight-update past-only calibration and diagnostics as Run 269. Rules are selected usingRun 269 only; later observations must not tune these branches.

## Verified execution and results

Completed 6 matched continuations on owned Spark, with loss states 448-1152 and 4,230 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0174389676. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `functional_entry_q50` | -20.94% | +13.56% | -41.36% | -82.03% |
| `functional_trunk_q75` | -8.44% | +35.32% | -38.58% | -71.44% |
| `functional_trunk_q50` | -22.85% | +4.36% | -39.04% | -85.49% |
| `global_half` | -16.18% | -12.97% | +49.15% | -56.86% |
| `global_quarter` | +1.40% | +39.18% | -41.36% | -99.49% |

The independent reduction checks 18 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 154.11 s, with peak allocated CUDA memory 592.2 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run272.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
