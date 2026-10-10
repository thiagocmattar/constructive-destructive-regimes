# Evidence supporting the proposed revision

The existing results support characterization of temporal geometry and its
response to a schedule intervention. They do not yet establish a scheduler,
a general stability criterion, or a causal benefit from reducing cancellation.

## Claims and their current status

| Proposed statement | Evidence | Status and boundary |
| --- | --- | --- |
| Raw gradients and Adam adaptive directions have substantially different adjacent and cumulative coherence. | [Run 002 observation](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/observations/001-adaptive-comparison.md) | Paired measurements along one AdamW trajectory. Adaptive direction excludes LR and weight decay. |
| Geometry changes systematically across the three model sizes under the original recipe. | [Analysis 002 replication](../../analyses/002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md) and [Analysis 003 audit](../../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md) | Three paired initialization/data-order seed blocks. Architecture changes with size; this does not isolate dimension or establish a scaling law. |
| Negative adjacent raw alignment can coexist with positive total interaction. | [Analysis 005 all-lag observation](../../analyses/005-2026-10-04-temporal-lag-profiles/observations/002-all-lag-interaction.md) | Exact aggregate identity on the recorded raw gradients; all nine phase/model windows have positive total interaction. One training seed. |
| Lag one alone is inadequate for cumulative interaction in these trajectories. | Same observation; [Analysis 003 replicated audit](../../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md) | Full-pass squared-resultant errors are 97.94–98.57% in the three temporal replays, and 97.90–98.57% in the nine-run original cohort. Distinct cohorts must remain labeled. |
| LR schedules affect local cancellation. | [Analysis 004 observation](../../analyses/004-2026-10-04-lr-schedule-intervention/observations/001-schedule-intervention.md) | Checkpoint-matched continuations at 14M/70M, seed 1235. LR also changes AdamW shrinkage. |
| Making alignment more positive improves learning. | The same intervention reports the opposite ordering of alignment and validation loss. | Unsupported as a monotone rule. Neither direction establishes mediation by alignment. |
| Adam's first moment reduces \(A\), increases \(B\), and closes the gap before preconditioning. | The cited figure measures \(\hat m/\left(\sqrt{\hat v}+\epsilon\right)\), not \(m\) or \(\hat m\). | Open hypothesis. Component measurements or calibrated reconstruction are needed. |
| Gradient compression enables useful temporal analysis. | [Analysis 005 method](../../analyses/005-2026-10-04-temporal-lag-profiles/METHOD.md) and calibration | Demonstrated for raw-gradient lag diagnostics. Accuracy for reconstructed momentum or feedback use requires its own checks. |

## What the cited Adam figure actually shows

The [raw and adaptive comparison](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/figures/raw_adaptive_comparison.pdf)
uses seed 1234, Pythia-14M. Its middle panels plot raw and adaptive \(A,B\)
separately because their scales differ. Solid lines are \(A\); dashed lines
are \(B\). Curve identity, rather than “the dotted line,” should define the
new prediction.

| Full-pass quantity | Raw gradient | Adam adaptive direction |
| --- | ---: | ---: |
| Median adjacent cosine | 0.007220 | 0.966250 |
| Final \(A\) | 319.486529 | 946630.439779 |
| Final \(B\) | 107.709080 | 667166.451433 |
| Final \(B/A\) | 0.337132 | 0.704780 |

Adaptive \(B/A\) reaches its minimum, 0.544333, at update 102 and ends at
0.704780. The final relative gap is 29.52%. This is a gap relative to aligned
vector accumulation, not an estimate that 29.52% of useful learning, gradient
energy, or compute was wasted.

All 712 updates enter these terminal values. The published plot shows updates
1–711 and excludes the final partial batch visually. It records no direct
momentum-only history. The adaptive/raw contrast combines clipping, temporal
averaging, bias correction, and coordinatewise preconditioning, so it cannot
attribute the entire difference to temporal averaging.

## Correct reading of the lag profiles

The [fixed origins figure](../../analyses/005-2026-10-04-temporal-lag-profiles/figures/lag_profiles_fixed_origins.pdf)
keeps the first 64 source updates fixed within each 128-update window for every
lag 1–64. It controls source coverage; it does not hold parameters, gradient
orientations, or target training states fixed.

The discussion remembered three negative initial lags. The saved
[results](../../analyses/005-2026-10-04-temporal-lag-profiles/results.json)
instead show:

| Middle window, updates 293–420 | Exact fixed-origin lag 1 | Projected lag 2 | Projected lag 3 |
| --- | ---: | ---: | ---: |
| 14M | -0.295602 | -0.127221 | +0.283970 |
| 31M | -0.307598 | -0.068147 | +0.215846 |
| 70M | -0.277714 | -0.081788 | +0.107943 |

Lag 2 and lag 3 are means across the four maps, taken from
the fixed_origin_profiles_by_projection field; lag 1 uses
fixed_origin_exact_lag_one_mean. The predeclared, all-within-window-pair
profiles also have positive lag-3 means: +0.2565, +0.2271, and +0.1240.
The fixed-origin control was added after inspecting early probes and remains
a sensitivity analysis.

The supported description is a **short-lag negative trough followed by a
positive rebound and a positive longer-lag background**. This is compatible
with short-horizon oscillatory structure. It does not identify a periodic
oscillator, its cause, or harmful motion across a loss valley. Lag one has a
prominent middle-window magnitude, but “lag one is always largest” is not
supported across phases; for example, early 14M lag 3 exceeds lag 1.

The negative trough also does not make the full window destructive under an
all-pair definition:

| Middle window | Exact adjacent interaction \(I_1\) | Exact all-pair interaction \(J\) |
| --- | ---: | ---: |
| 14M | -10.6946 | +169.0948 |
| 31M | -10.0077 | +159.6306 |
| 70M | -9.9227 | +127.2076 |

These norm-weighted interactions are different from equally weighted mean
cosines. Many positive longer-lag contributions outweigh the negative local
contributions. See the [all-lag figure](../../analyses/005-2026-10-04-temporal-lag-profiles/figures/cumulative_lag_interaction_all_lags.pdf).

## The schedule intervention is a central result

[Analysis 004](../../analyses/004-2026-10-04-lr-schedule-intervention/REPORT.md)
branches the same update-356 state into original decay, held midpoint LR,
and accelerated decay. It does not intervene near update 100.

| Model | Schedule | Negative intensity \(D\) | Final validation loss |
| --- | --- | ---: | ---: |
| 14M | Original | 0.0681 | 5.1820 |
| 14M | Held | 0.3620 | 5.0674 |
| 14M | Accelerated | 0.0222 | 5.2176 |
| 70M | Original | 0.1591 | 4.0559 |
| 70M | Held | 0.3201 | 3.9846 |
| 70M | Accelerated | 0.0981 | 4.1042 |

Here \(D=\operatorname{mean}\max(-\rho_t,0)\) over affected pairs 358–712.
Validation covers all 500 documents and 693,329 supervised targets, including
the tail. Faster decay brings the declared positive-smoothed criterion forward
by 65/101 updates at 14M/70M, but gives the highest terminal validation loss.
Positive recovery at 70M is temporary.

This intervention supports sensitivity of raw-gradient cancellation to the
schedule. It motivates testing a relationship between geometry and the size
of a useful step, but it does not establish that negative means “too large”
or positive means “too small.” The primary comparison is one seed at two sizes,
conditional on a trained prefix, with step scale and decay coupled.

## Coverage and uncertainty to retain in the paper

The original scale cohort has three seeds per size. The temporal-profile
cohort has one seed, 1235, per size; four projections are not four training
replications. All use random initialization and a full MiniPile pass under the
recorded recipe. Raw gradients are measured after token-weighted accumulation
and unscaling, before clipping and AdamW.

The temporal replays match pinned initial tensors, data and configuration, but
are not bitwise replicas of the earlier runs. GPU differences and trajectory
sensitivity preclude splicing their histories into Run 002. Phase profiles
describe the replay that produced them.

Four fixed CountSketch maps of 16,384 buckets record every accepted update.
The FP32 history contains 186,646,528 data bytes per model. Exact raw-pair
calibration gives whole-history cosine RMSE 0.00380–0.00444 and maximum observed
error 0.01849; exact aggregates establish the all-pair total independently.
These are empirical calibration results, not uniform error bounds. Map bands
describe projection sensitivity, not uncertainty across training seeds.

No current result measures actual AdamW displacement at every update,
separately identifies per-sample energy and intra-batch alignment, or establishes
forgetting, formal stability, or a causal effect of interference on learning.
