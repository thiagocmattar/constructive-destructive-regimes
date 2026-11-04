# Spark fixed-step intervention responses

Status: running. Completed primary branches: 7/30.

All interventions start from the same saved state1024, with warm Adam moments. Contemporary unchanged Spark continuations are the controls. The historical before period uses prior hardware and is explicitly separated at the intervention boundary.

Ten primary conditions test fixed LR reductions, cosine floors, global clipping and decoupled weight decay; optional cosine durations depend only on remaining time. One selected seed/state;64 new full-objective updates measure short-term responses.

| Case | Intervention | Mean loss change vs control | Final loss change vs control | Clipped updates |
| --- | --- | ---: | ---: | ---: |
| mlp | unchanged | +0.000% | +0.000% | 0/64 |
| mlp | fixed_half | -46.824% | -38.256% | 0/64 |
| mlp | fixed_quarter | -45.766% | -35.567% | 0/64 |
| resnet | unchanged | +0.000% | +0.000% | 0/64 |
| resnet | fixed_half | +4.838% | +9.696% | 0/64 |
| transformer | unchanged | +0.000% | +0.000% | 0/64 |
| transformer | fixed_half | +3.234% | +4.855% | 0/64 |

Negative percentages mean lower unregularized training loss. Complete held-out endpoints are descriptive; no tuning or generalization claim. Queued, failed and interrupted branches are retained in results.json.

[Numeric comparison](comparison.csv), [aligned values](aligned_values.csv), and [status/verification](results.json). Main figures have three model/dataset columns, solid curves, independent Y ranges and five rows: loss, raw rho, raw gradient norm, projected raw change and LR. No smoothing or unreported clipping of curves. Weight-decay companions show post-intervention parameter, actual-increment and decay-increment norms; no unrecorded historical weight norms are invented.

Run222 actual updates: 499 main + 12 qualification; recorded full gradient calls: 527. Separate Run221 mechanics:9 updates and3 zero-update warm-parent gradient calls. Cloud spendUSD0; owned Spark.

Final interpretation and5-10 factual findings are added after terminal retrieval and numerical/visual verification. The failed Run221 cross-GPU gradient gate remains part of the evidence.

Available response figures:

- [fixed_half.pdf](figures/fixed_half.pdf)
