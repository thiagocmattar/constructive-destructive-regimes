# Run 275: matched hidden-layer increment norms

H048 / Analysis 065. Compare unchanged Adam with a scalar-per-layer control whose two hidden affine increment norms match Run 272 functional_trunk_q50 at each update. This isolates within-layer selection from norm scheduling. All gradients, Adam moments, objective and base LR are unchanged; the output affine increment remains native. Scalar factors may exceed 1. This retrospective counterfactual uses the selected arm's schedule and is not an online policy. Report early 448-800 and held-forward 801-1152 separately.

## Verified execution and results

Completed 2 matched continuations on owned Spark, with loss states 448-1152 and 1,410 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0174389676. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `matched_trunk_q50` | -3.43% | +76.61% | -22.89% | +86.90% |

The independent reduction checks 6 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 27.20 s, with peak allocated CUDA memory 553.1 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run275.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
