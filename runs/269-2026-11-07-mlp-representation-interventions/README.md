# Run 269: mlp representation-informed interventions

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

Completed 10 matched continuations on owned Spark, with loss states 448-800 and 3,530 applied updates including the terminal outgoing diagnostics. The unchanged Adam mean loss is 0.0228078201. All native gradients, moments and clocks advance; each intervention is compared with that run's unchanged control.

Percent changes below use the full loss window; negative means lower. Tail means the last 32 loss states. PV is total positive adjacent loss variation.

| Arm | Mean loss | Tail loss | Maximum loss | PV |
|---|---:|---:|---:|---:|
| `control` | +0.00% | +0.00% | +0.00% | +0.00% |
| `discard_entry` | +1.45% | +41.17% | -31.24% | -9.02% |
| `cap_entry` | -2.19% | +10.27% | -32.77% | -42.35% |
| `functional_entry` | -8.26% | +8.50% | -37.91% | -69.50% |
| `block_shock` | +612.39% | +1847.50% | +5086.91% | +1603.65% |
| `global_half` | -18.71% | -14.93% | -41.36% | -99.65% |
| `functional_entry_q50` | -20.31% | -10.42% | -41.36% | -86.49% |
| `global_shock` | +531.12% | +843.58% | +3686.65% | +1085.02% |
| `block_secant` | +57.94% | +10.79% | +245.20% | +505.53% |
| `first_quarter` | -7.06% | +15.79% | -41.36% | -99.62% |

The independent reduction checks 30 dense audit states, past-only calibration, gate recurrences, FP32 placement, optimizer clocks and exact within-worker fork identity. Same-hardware replay verifies complete-probe representation RMS/cosines. Source and archive/per-file SHA-256 checks pass; finished Spark processes and logs are verified separately.

Summed continuation durations are 79.37 s, with peak allocated CUDA memory 588.1 MB. This excludes prefix, mechanics, replay and transfers; concurrent Spark jobs prevent treating these durations as isolated throughput benchmarks. Cloud rental cost for this run is zero. Raw evidence remains under `artifacts/` locally and on owned Spark.

Numerical record: [verified comparison](../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_comparison_mlp.json). See [methods and transfer results](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) and [MLP mechanism/negative controls](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/009-what-the-negative-controls-show.md). Scientific inputs remain frozen; this result section adds no new run or policy.
