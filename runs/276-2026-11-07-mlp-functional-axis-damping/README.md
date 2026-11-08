# Run 276: causal functional-axis damping

H048 / Analysis 065. A cheap MLP-only exploratory test directly targets motion along the preceding first-projection change. Given native local responseZ=X dW+db and previous actual responseV, beta=<Z,V>/<V,V>. If beta<0, replace it by 0, beta/2, or max(beta,-tau/RMS(V)), where tau is the eight-update prefix-q50 functional threshold. Apply the same subtraction in weight/bias space, which is exact for the first affine projection on fixed inputs. Orthogonal response remains unchanged up to FP32 storage rounding. This uses one past direction rather than a window-fitted SVD basis. No claim of a fixed dominant axis, and no future data. Compare with unchanged Adam through 1152; all layers remain trainable.

## Verified execution and results

Completed 4 matched continuations on owned Spark, with loss states 448-1152 and 2,820 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0174389676. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `axis_remove` | +171.71% | -35.41% | +1631.67% | +375.29% |
| `axis_half` | +79.61% | -19.83% | +584.18% | +213.11% |
| `axis_cap` | +171.47% | +54.50% | +1703.47% | +485.08% |

The independent reduction checks 12 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 409.19 s, with peak allocated CUDA memory 570.2 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_run276.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
