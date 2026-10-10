# Observation 002: LR decay and late recovery of raw-gradient alignment

## Question, contract, and provenance

The user asks whether the declining LR can explain the late return toward
positive cosine, especially the earlier recovery of 14M, and therefore part of
the full-pass size difference. H004 records this as a plausible mechanism to
investigate, not an identified causal effect. The 14M full-pass cosine median
is numerically higher (0.007220) than 31M (-0.086252) and 70M (-0.160566);
the user's substantive observation is its earlier recovery and lower negativity.

Use the same completed, matched seed 1234 cohort and raw-gradient contract as
[Observation 001](001-raw-gradient-regimes.md): Run 002 attempt 02, Run 003 attempt 01,
Run 004 attempt 01. Scientific inputs and scalar-event hashes are rechecked against
Analysis 001. `../schedule_results.json` records source paths/hashes, configuration,
executed recipe/training source hashes, coverage, phases, and verification.
`../schedule_data.csv` retains all 2136 boundaries including the partial final batch.
The executed sources are used to establish the capture/scheduler order.

Raw gradients at boundary t are captured before the optimizer acts at that
boundary. The LR at t-1 produced the transition to the parameters used by g_t,
so the preceding LR is the timing-aligned descriptive covariate for rho_t.
Neither preceding nor current LR is randomized. Positive rescaling of two
nonzero vectors leaves their cosine unchanged: LR can affect this raw diagnostic
through parameter/optimizer-state evolution, not by directly shrinking its angle.

Matched factors remain cache/order, seeds, tokens, updates, batch, precision,
initialization recipe, AdamW configuration and nominal LR. Varying factors remain
model architecture/size and their resulting trajectories. There is no between-run
schedule variation. Phase statistics use unsmoothed finite cosines with equal
pair weights and supervised-token weighted loss. No p-values or step-based
replicate confidence intervals are computed.

## Recorded schedule and recovery

All three logged LR sequences are exactly equal at every boundary. The configured
peak is 0.001, warmup fraction 0.01 and minimum 0.0001. The frozen GPT-NeoX pre-step
recipe reproduces all 712 values to maximum absolute difference 1.09e-19 when
recomputed on Windows (floating-point platform rounding). The first LR is zero;
the near-peak occurs at update 9. Cosine decay is clipped at the floor from
optimizer update 575 onward. Thus pairs 576-711 all follow a step with LR 0.0001.

An exploratory recovery marker is the first center after midpoint 356 starting
31 consecutive strictly positive centered 31-update cosine means, restricted to
complete windows. It is retrospective, includes future observations, and is
neither a permanent recovery nor a detected dynamical phase boundary.

| Model | First recovery center | Preceding LR | 15-update smoother | 61-update smoother |
| --- | ---: | ---: | ---: | ---: |
| 14M | 476 | 0.00026502 | 497 | 485 |
| 31M | 500 | 0.00021969 | 496 | 499 |
| 70M | 541 | 0.00014977 | No qualifying interval | 542 |

The hold length remains 31 centers in each sensitivity. The 14M/31M ordering
reverses slightly at width 15, and 70M fails that criterion at width 15. These
are illustrative descriptions of the original 31-update figure, not robust
LR thresholds. Exact recovery locations depend on the reduction.

## Common phases and the constant-LR segment

| Model | Median rho, updates 179-356 | Median rho, updates 357-500 | Median rho, updates 576-711 | Negative pairs, updates 576-711 |
| --- | ---: | ---: | ---: | ---: |
| 14M | -0.287772 | -0.159986 | +0.360203 | 2.94% |
| 31M | -0.321731 | -0.146887 | +0.155987 | 19.12% |
| 70M | -0.282898 | -0.218456 | -0.003427 | 52.21% |

The early middle-phase medians lack the full-pass monotonic size ordering.
In the late fixed-LR segment, 14M and 31M are clearly more positive than 70M.
The prefix 1-534 cosine medians are -0.152661, -0.171116, -0.217429; the complete
medians become +0.007220, -0.086252, -0.160566. The late segment changes the
global comparison substantially. Medians are not additive, so this is not a
numerical decomposition of the full median into phase contributions.

Within 70M at the same LR 0.0001, mean cosine is +0.038944 over 576-620 and
-0.110487 over 621-665, while mean training loss falls 4.029922 -> 4.008266.
The latter segment has 68.89% negative pairs. Thus constant nominal LR can
coexist with renewed negative alignment and lower subsequent batch losses.
This rules out a memoryless description rho=f(LR) for these records, not a
history-dependent effect of the schedule on the trajectory. Batch loss is not
reevaluation of one fixed objective; no causal loss effect is inferred.

## Interpretation and next discriminating experiment

The decay/recovery timing is consistent with the user's explanation. It does
not identify the causal effect: LR, elapsed training, processed tokens, changing
parameters, optimizer moments and batch exposure co-evolve. The same nominal LR
also does not imply the same adaptive update size or curvature-normalized step
at different widths. Effective curvature and actual parameter updates were not
measured here. Current data support a size-dependent response to the common
training recipe; they do not separate schedule effects from intrinsic size effects.

A fixed quadratic gives a useful mechanism check. For L(theta)=lambda*theta^2/2,
plain GD satisfies g_(t+1)=(1-eta*lambda)g_t. With 1<eta*lambda<2, gradients
alternate while loss still contracts. Reducing eta below 1/lambda makes them
same-sign. This shows that LR decay can cause sign recovery without a new minimum
or a change from divergence to convergence. It is a toy example, not evidence
that Pythia satisfies that mechanism. AdamW's preconditioner, moments, decay and
changing mini-batches prevent direct substitution of the scalar threshold.

The restricted interpretation is **recovery of local raw-gradient alignment
under a declining LR, with different responses across sizes**. "More convergent"
requires a separate criterion for fixed-objective loss, stationarity or trajectory
sensitivity; positive rho alone does not establish it. The existing hypotheses
and earlier results are retained.

A future discriminating experiment would fork the same pre-recovery checkpoint
with identical model, optimizer, scaler, random states and future data order:
continue decay in one branch and hold LR in another, retaining an original-schedule
replay control. Branches with changed schedules require new run IDs and frozen
inputs. This is a proposed intervention only; it was not implemented, and no new
workers or training were initiated by this extension.

Primary literature context: [Xing et al., *A Walk with SGD*](https://arxiv.org/abs/1802.08770)
describes LR-dependent bouncing across valley-like regions in its own SGD
experiments. That motivates the plausibility of a step-size mechanism, without
establishing it in this AdamW/Pythia cohort or measuring sharpness here.

## Figure caption and verification

`schedule_and_raw_cosine.pdf`: Identical preceding-update LR (left) and adjacent
whole-model raw-gradient cosine (right) along matched seed 1234 Pythia 14M/31M/70M
MiniPile trajectories. Cosine lines use centered 31-update means with shrunken
edge windows and null omission; LR is unsmoothed. Shading starts at boundary 576,
the first gradient pair preceded by a floor-LR step. Display includes 1-711,
while the smaller boundary 712 remains in the data. Colors and line styles
identify size. No uncertainty band, correlation test or causal estimate is shown.

Generated by `../03_schedule.py`. Three mathematical checks verify positive
rescaling invariance, a contracting quadratic whose alignment changes under LR
reduction, and recovery detection excluding brief excursions/incomplete windows.
The recorded schedules, frozen-source reproduction, data coverage, PDF text bounds,
visual render and source preservation were checked. Results retain figure/script
hashes. This is one seed per size; recovery definitions and smoothing are
exploratory. The running replications, Analysis 002 and manuscript assets are untouched.
