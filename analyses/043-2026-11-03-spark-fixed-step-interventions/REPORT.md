# Spark fixed-step intervention responses

Status: completed. Completed primary branches: 30/30.

All interventions start from the same saved state1024, with warm Adam moments. Contemporary unchanged Spark continuations are the controls. The historical before period uses prior hardware and is explicitly separated at the intervention boundary.

Ten primary conditions test fixed LR reductions, cosine floors, global clipping and decoupled weight decay; optional cosine durations depend only on remaining time. One selected seed/state;64 new full-objective updates measure short-term responses.

| Case | Intervention | Mean loss change vs control | Final loss change vs control | Clipped updates |
| --- | --- | ---: | ---: | ---: |
| mlp | unchanged | +0.000% | +0.000% | 0/64 |
| mlp | fixed_half | -46.824% | -38.256% | 0/64 |
| mlp | fixed_quarter | -45.766% | -35.567% | 0/64 |
| mlp | cosine32_floor0p5 | -45.876% | -39.306% | 0/64 |
| mlp | cosine32_floor0p25 | -45.722% | -37.338% | 0/64 |
| mlp | cosine32_floor0p1 | -45.519% | -36.015% | 0/64 |
| mlp | clip_median_x1 | -16.585% | +21.956% | 15/64 |
| mlp | clip_median_x2 | +0.000% | +0.000% | 0/64 |
| mlp | adamw_0p01 | +0.182% | +1.869% | 0/64 |
| mlp | adamw_0p1 | +1.902% | +14.659% | 0/64 |
| mlp | cosine8_floor0p25 | -46.666% | -36.025% | 0/64 |
| mlp | cosine64_floor0p25 | -44.907% | -39.218% | 0/64 |
| resnet | unchanged | +0.000% | +0.000% | 0/64 |
| resnet | fixed_half | +4.838% | +9.696% | 0/64 |
| resnet | fixed_quarter | +7.436% | +15.114% | 0/64 |
| resnet | cosine32_floor0p5 | +2.641% | +7.078% | 0/64 |
| resnet | cosine32_floor0p25 | +4.031% | +10.911% | 0/64 |
| resnet | cosine32_floor0p1 | +4.887% | +13.312% | 0/64 |
| resnet | clip_median_x1 | +0.000% | +0.000% | 0/64 |
| resnet | clip_median_x2 | +0.000% | +0.000% | 0/64 |
| resnet | adamw_0p01 | +0.615% | +1.228% | 0/64 |
| resnet | adamw_0p1 | +6.354% | +12.872% | 0/64 |
| resnet | cosine8_floor0p25 | +6.362% | +13.944% | 0/64 |
| resnet | cosine64_floor0p25 | +2.022% | +7.046% | 0/64 |
| transformer | unchanged | +0.000% | +0.000% | 0/64 |
| transformer | fixed_half | +3.234% | +4.855% | 0/64 |
| transformer | fixed_quarter | +5.922% | +9.099% | 0/64 |
| transformer | cosine32_floor0p5 | +1.582% | +3.449% | 0/64 |
| transformer | cosine32_floor0p25 | +2.687% | +6.129% | 0/64 |
| transformer | cosine32_floor0p1 | +3.492% | +8.179% | 0/64 |
| transformer | clip_median_x1 | -0.058% | -0.178% | 3/64 |
| transformer | clip_median_x2 | +0.000% | +0.000% | 0/64 |
| transformer | adamw_0p01 | +0.025% | +0.015% | 0/64 |
| transformer | adamw_0p1 | +0.656% | +1.159% | 0/64 |
| transformer | cosine8_floor0p25 | +4.862% | +8.338% | 0/64 |
| transformer | cosine64_floor0p25 | +1.133% | +3.477% | 0/64 |

Negative percentages mean lower unregularized training loss. Complete held-out endpoints are descriptive; no tuning or generalization claim. Queued, failed and interrupted branches are retained in results.json.

[Numeric comparison](comparison.csv), [aligned values](aligned_values.csv), and [status/verification](results.json). Main figures have three model/dataset columns, solid curves, independent Y ranges and five rows: loss, raw rho, raw gradient norm, projected raw change and LR. No smoothing or unreported clipping of curves. Weight-decay companions show post-intervention parameter, actual-increment and decay-increment norms; no unrecorded historical weight norms are invented.

Run222 actual updates: 2304 main + 12 qualification; recorded full gradient calls: 2361. Separate Run221 mechanics:9 updates and3 zero-update warm-parent gradient calls. Cloud spendUSD0; owned Spark.

[Eight verified findings and full interpretation](FINDINGS.md); [checkpoint and provenance checks](verification.json). The failed Run221 cross-GPU gradient gate remains part of the evidence.

Available response figures:

- [adamw_0p01.pdf](figures/adamw_0p01.pdf)
- [adamw_0p01_increments.pdf](figures/adamw_0p01_increments.pdf)
- [adamw_0p1.pdf](figures/adamw_0p1.pdf)
- [adamw_0p1_increments.pdf](figures/adamw_0p1_increments.pdf)
- [clip_median_x1.pdf](figures/clip_median_x1.pdf)
- [clip_median_x2.pdf](figures/clip_median_x2.pdf)
- [cosine32_floor0p1.pdf](figures/cosine32_floor0p1.pdf)
- [cosine32_floor0p25.pdf](figures/cosine32_floor0p25.pdf)
- [cosine32_floor0p5.pdf](figures/cosine32_floor0p5.pdf)
- [cosine64_floor0p25.pdf](figures/cosine64_floor0p25.pdf)
- [cosine8_floor0p25.pdf](figures/cosine8_floor0p25.pdf)
- [fixed_half.pdf](figures/fixed_half.pdf)
- [fixed_quarter.pdf](figures/fixed_quarter.pdf)
