# Run 279: slow recovery of functional damping

H048 / Analysis 065. Test whether immediate release of a global movement cap caused the poor Run 278 trajectory. Reuse the same eight-update q50 calibration, actual probe movement checks, full objective and unchanged Adam states. The global factor can decrease immediately but increase at most 1% per update. No future measurements or loss acceptance criterion. Both output-only and all-tracked-representation variants compare with a fresh exact unchanged control over 448-1152.

## Verified execution and results

Completed 3 matched continuations on owned Spark, with loss states 448-1152 and 2,115 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0174389676. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `output_budget_slow` | -19.62% | -8.17% | -26.21% | -43.68% |
| `representation_budget_slow` | -19.70% | +7.75% | -26.21% | -42.48% |

The independent reduction checks 9 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 44.69 s, with peak allocated CUDA memory 553.1 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run279.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
