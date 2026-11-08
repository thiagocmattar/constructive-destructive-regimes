# Repository evidence review

**Iteration 2 — 8 November 2026. Read-only scientific review; consolidation document only.**

This review fills the author's existing six-section paper and contribution
components C1–C8 in [PLAN.md](../PLAN.md). It covers the full Analysis001–065
range, including the early scaling, controlled geometry, sampling and temporal
studies. Tracing and intervention results extend that foundation within Section V.
They do not replace the GD/SGD argument or select a different manuscript structure.

## Source authority and review coverage

The [author inputs](../human/) and active manuscript determine the scientific
direction and organization. Executed configurations, retained measurements and
their provenance determine what each experiment establishes. Analysis observations,
reports, figures and scripts explain the reductions. Registries and historical
plans locate work; they are not sufficient evidence that a task remains unfinished.
The [research index](../../../research/INDEX.md) is an entry point, not a substitute
for the owning analysis.

This review surveyed the actual README/report records across **all 65 numbered
analyses**, with detailed reading of the observations and measurement contracts
for the results discussed below. It also checked the active manuscript inputs,
the rollback record, and the [definitions](../../../research/DEFINITIONS.md),
[methods](../../../research/METHODS.md), [metrics](../../../research/METRICS.md)
and [tracing contract](../../../research/INSTABILITY_TRACING.md). The inventory
below distinguishes completed experiments, reanalyses, presentations, plans and
engineering checks. This was not a new numerical reproduction of every run or
a fresh integrity audit of every binary artifact; existing verification receipts
remain attributable to their owning records.

## Active manuscript and review history

The latest entry in [manuscript/README.md](../../README.md) is authoritative:
on 6 November the active TeX and bibliography were restored to
`295b7590c2d17ee80419da1f71d9486e0ae919c0`, preserving the author's introduction
edits from `f9131dd`. The recorded rebuilt draft has 13 pages. Later analyses,
figures and revision folders were preserved, but their prose is not automatically
part of the active paper. The older restoration entry at the top of
[research/MANUSCRIPT.md](../../../research/MANUSCRIPT.md) describes an earlier
state and must not override the later rollback.

The [compilation root](../../draft/bare_jrnl.tex) uses the introduction, related
works, methods, GD experiments, SGD-noise subsection, Adam diagnostics, conclusion
and two appendices. In particular, file names are not section numbers: the SGD
file contributes to Section IV. Legacy `04-results.tex` and `06-appendix.tex` are
not active inputs. Their older per-sample/batch algebra can inform the background,
but it must not be described as an already active empirical decomposition. A
batch-mean gradient alone cannot separately recover per-sample energy and
intra-batch alignment. The active argument principally follows temporal
interactions between successive batch/full-objective gradients.

The current main empirical assets, included through
[04-controlled-figures.tex](../../draft/sections/04-controlled-figures.tex), are:

| Current asset | Evidence retained in the six-section plan |
|---|---|
| [GD loss, projected change and spectral curvature](../../draft/figures/analysis024/gd_loss_projected_change_spectral_clean.pdf) | Section IV convergence/curvature; Analysis024. |
| [Reversal, norm and retention](../../draft/figures/analysis024/gd_reversal_norm_retention.pdf) | Section IV destructive regimes, parameter movement and R; Analysis024. |
| [Exact backward lags](../../draft/figures/analysis035/gd_backward_compact.pdf) | Section IV temporal interactions; Analysis035. |
| [Batch-size comparison](../../draft/figures/analysis037/sgd_batch_size_compact.pdf) | Section IV stochastic sampling and moving paths; Analysis037. |

The [5 November readiness review](../../2026-11-05-submission-readiness/README.md)
assessed the then-current 18-page draft, not the restored 13-page baseline.
Its objections remain useful questions: define the sense of stability, justify
the interpretation of parameter norm, test the neglect of nonadjacent terms,
attribute standard norm/secant identities accurately, and connect the diagnostic
to an explanatory contribution. The old covariance-cost and energy-distance
criticisms must be checked against the formulations actually retained. The
supplied [previous-submission records](../../draft/previous-submission/) are
preserved originals, not instructions to reapply historical edits. The decision
letter's restriction on revised resubmission is a separate submission-route
item in PLAN/W02; this review does not change the author's TNNLS objective.

## I. Introduction — learning dynamics and the contribution chain

**C1–C8 have an existing evidence chain.** The starting point is the exact
decomposition of accumulated SGD displacement into individual step energies
and pair interactions. Adjacent constructive/destructive behavior gives an
interpretable local characterization. The curvature connection, amplification,
sampling and longer-lag measurements explain how that characterization relates
to loss and accumulated motion. The introduction can present these relationships
as the foundation, then explain how Adam, representation tracing and interventions
extend the same learning-dynamics question.

The early Pythia work already provides motivation at language-model scale.
[002](../../../analyses/002-2026-10-04-scale-seed-replication/README.md) compares
14M, 31M and 70M models over 712 updates with seed blocks 1234/1235/1236;
[003](../../../analyses/003-2026-10-04-raw-gradient-regimes/README.md) distinguishes
negative-event frequency from conditional magnitude. Larger models improve loss
while raw adjacent gradients become negative more frequently. This is useful
evidence that local opposition and successful learning coexist. It is an observed
three-size pattern, not a capacity law: architecture dimensions vary, seeds also
change data order, and hardware differs across some blocks.

[004](../../../analyses/004-2026-10-04-lr-schedule-intervention/REPORT.md) gives a
direct learning-rate example: from a common state 356, faster decay makes raw
alignment positive earlier, while holding LR gives the best validation loss in
both tested sizes. [025](../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md)
adds a controlled optimizer contribution: removing coordinate scaling while
preserving the parent adaptive-step norm schedule worsens all 18 short branches.
These findings support a learning interpretation that includes magnitude,
direction and optimizer transformation. They do not make alignment positivity
the objective of training.

## II. Stability Analysis of SGD — evidence that informs positioning

The paper can keep its current curvature, stochasticity, edge-of-stability and
practical-model progression. The repository supplies several concrete distinctions
for that discussion:

- [019](../../../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md) has four GD and
  four Adam MLP trajectories over 20,000 steps, ordinary Hessian spectra and
  preconditioned Adam spectra. The three larger GD rates approach adjacent
  cosine near −0.999 and eta-times-leading-curvature near 2.07 while attaining
  lower training loss. Ordinary and preconditioned Adam spectral quantities
  differ substantially; they are not interchangeable stability thresholds.
- [020](../../../analyses/020-2026-10-14-eos-multitask/README.md) and
  [027](../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) distinguish
  sampling at a fixed parameter state from stochastic motion across states.
  This supplies an empirical basis for treating noise-induced misalignment
  separately from curvature-induced reversal.
- [032–035](../../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md)
  directly test the temporal locality assumptions underlying an adjacent
  characterization. Reversal may alternate over many lags; nonadjacent terms
  cannot simply be discarded without specifying the approximation.
- [017](../../../analyses/017-2026-10-12-rosenbrock-kappa-identity/README.md)
  and [018](../../../analyses/018-2026-10-12-rosenbrock-kappa-noise/README.md)
  provide small exact controls for the secant/projection interpretation. These
  help attribute the classical mathematical ingredients while explaining the
  framework's diagnostic use.

These are local empirical/formal supports for the literature positioning in
[ARGUMENT_MAP.md](ARGUMENT_MAP.md), not claims that the repository alone proves
novelty. The current introduction and II remain the organizing context.

## III. Constructive and Destructive Regimes — formulation and direct checks

### Directional curvature and inexpensive estimator: C1/C2

Keep the current update decomposition, directional relation, path-average
identity and stated approximation/error conditions. The useful contribution is
the relationship between gradient response and the learning trajectory, with a
cheap observable from consecutive gradients. The observation under Adam is a
projection along the optimizer-selected path; it is not automatically an
ordinary Hessian eigenvalue.

The low-dimensional [016–018 controls](../../../analyses/017-2026-10-12-rosenbrock-kappa-identity/README.md)
use exact float64 Rosenbrock derivatives and explicit injected noise. More
directly relevant to neural SGD, [027](../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md)
contains **1,188 state-matched HVP pairs** on the full 5,000-example objective.
At batch 32, the raw proxy's median absolute discrepancy is 0.7467; after the
explicitly measured noise correction it is 0.0030, with 90th percentile 0.0143.
The corresponding corrected medians at batches 128/512/2048 are
0.0033/0.0034/0.0025. These are independently computed derivative comparisons,
not merely an algebraic self-check. The correction uses full gradients as an
offline reference; it is not itself advertised as a free online quantity.

### Sampling and batch size: C4

[020](../../../analyses/020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md)
has 36 fixed-state conditions across three models, three seeds and four batch
sizes, with 147,456 sampled draws and no optimizer updates. Full-batch controls
have rho and R equal to one and zero corrected gradient change. The smallest
batch produces different noise levels across architectures, so batch size alone
does not imply a universal observed cosine.

[027](../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md) expands this to
three seeds, states 0/1000/2000, batches 32/128/512/2048/5000, 184,320 frozen
draws, 297 moving forks and the HVP checks above. Its variance/noise-projection
checks agree within 0.50%/0.74%; the approximate cosine has maximum absolute
error 0.00745, with detectable small bias in 21 of 36 bootstrap comparisons.
This supports the decomposition and gives a precise domain for the approximation.

### Normalized displacement and temporal interactions: C1/C5

[024](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md)
and [056](../../../analyses/056-2026-11-05-gd-paired-displacement/observations/001-paired-displacement.md)
show positive final all-pair interaction in all nine GD cases even though seven
have negative cumulative adjacent interaction. This directly supports the
distinction between local destructive behavior and accumulated parameter motion.
For the three MLP rates, absolute displacement grows approximately
0.951/3.139/4.763 while path length grows 1.198/64.361/185.542; lower retention R
therefore coexists with greater net displacement. Parameter norm, displacement
and retention answer different questions.

Analysis056 also supplies an exact paired C/D factorization and zero-cross-term
reference. Those calculations are available supporting evidence; the author's
rollback removed that notation from the active draft. Its restoration is not a
prerequisite for using the all-pair checks or explaining R in the current form.

### Loss decrease and amplified excursions: C3

[024's finite-step audit](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md)
already evaluates 26,991 transitions. The local loss approximation is informative
but not exact: sign disagreements vary by model/rate, reaching 1,325 of 2,999
transitions in the highest-rate Transformer case. A loss-change threshold of
1e−5 leaves 1,239 disagreements among 2,837 eligible transitions. These retained
checks can be summarized compactly alongside the analytical relation. They
avoid claiming that the approximate formula alone determines every finite-step
loss change. Any quantity using the next gradient is retrospective with respect
to the preceding update.

For Adam, [057](../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md)
finds a descending first-order actual-step contribution in all 794 rise-phase
updates of 129 recovered events above 10%. [062](../../../analyses/062-2026-11-07-mlp-gradient-amplification/README.md)
and [065](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md) supply
functional accounting for the MLP's squared loss: baseline rises occur with
negative linear contribution and a larger positive quadratic contribution.
This supports reversal-plus-amplification interpretation while distinguishing
gradient response, actual movement and finite-step loss.

## IV. Empirical Characterization — preserve the four existing stages

**Convergence and curvature.** [024](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md)
contains nine 3,000-update full-objective GD cases: MLP rates 0.004/0.02/0.04,
ResNet 0.025/0.05/0.1 and Transformer 0.05/0.1/0.2. The highest rate attains
the lowest final training loss in each model. The objective covers 5,000 examples
for MLP, 50,000 for ResNet and 2,088,600 tokens for Transformer. Spectral
measurements cover the full MLP objective but subsets for the larger models
(5,000 ResNet examples and 2,500 of 59,680 Transformer sequences). One
initialization per case supports a controlled comparison, not seed uncertainty.

**Destructive regimes, norm and R.** Retain the second current figure and the
exact accumulation checks above. The evidence supports productive oscillatory
learning together with a large difference between path length and net displacement. It does not equate a
negative cosine, lower parameter norm or small R with loss divergence.

**Temporal lags.** [035](../../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md)
has 18 GD/Adam profiles from saved states 500/2000/3000, using exact backward
lags 1–256 at endpoints 756/2256/3256. The current compact GD figure uses endpoint
2256. Earlier [032](../../../analyses/032-2026-11-01-eos-exact-lag10/REPORT.md)
found positive even/negative odd lags in all 17 reversal profiles;
[033](../../../analyses/033-2026-11-01-eos-long-lags/REPORT.md) extended selected
profiles to lag 2048 and found persistence and rebounds. These results support
the importance of longer-range organization. Fresh continuations are not assumed
bitwise identical to every historical trajectory, and an anchored profile is
not an all-pair cumulative sum.

**Minibatch effects.** [037](../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md)
already supplies the cross-architecture extension: 348 new forks plus 198 MLP
forks, totaling 546 trajectories, three seeds, eight stochastic repeats, and
200 updates from states 0/2000 with full-GD controls. At trained states,
small-batch/full-GD adjacent cosine is approximately −0.384/−0.991 for MLP,
−0.612/−0.967 for ResNet and −0.258/−0.993 for Transformer. Transformer batch 8
has similar cosine around −0.259 at the two states but loss changes −3.505 and
+0.0643, respectively, with the sign contrast present across all three
initialization means. Thus the same alignment can accompany different progress.
ResNet uses intact 1,000-example BatchNorm groups; Transformer sampling uses
the fixed-total-token denominator. Equal update counts are not equal token
budgets, and the recorded seed-2 ResNet runtime difference remains a local caveat.

## V. Raw-Gradient Diagnostics under Adam — characterization, tracing, action

### Adaptive-path interpretation and high-rate characterization: C6

[028](../../../analyses/028-2026-10-30-adam-diagnostics/README.md) is the historical
synthesis; [029](../../../analyses/029-2026-10-30-pythia-adam-diagnostics/REPORT.md)
recovers incoming-gradient/prior-moment alignment from 4,977 scalar-log pairs
and verifies the clipped-norm recurrence. The streams must remain distinct:
raw gradient, clipped optimizer input, moment, adaptive direction, native proposal
and actually applied parameter increment.

[036](../../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md)
provides nine controlled full-objective Adam arms, three rates per model, seed 0,
over 1,024 updates. Its loss-defined event population contains 812 candidates,
806 recovered events and six censored cases; 129 recovered rises exceed 10%,
and 127 have defined pre-rise cosine. [055](../../../analyses/055-2026-11-05-adam-directional-amplification/observations/002-direction-context.md)
retains the full quiet/event-eligible population, and 057 provides complete-event
loss accounting. These sources can support simplified high-rate main displays
without silently replacing a population with selected peaks.

[058](../../../analyses/058-2026-11-06-intervention-consolidation/README.md) already
consolidates the requested figures: 00 high-rate overview, 01 half-LR, 02 floors
and return, 03 restoration, 04 two-stage schedules, 05 clipping, 06 Pythia
schedules, 07 optimizer diagnostics and 08 spike zoom. This is presentation of
existing runs, not an additional training study. The requested peak table still
needs explicit state/update availability: pre-update observations, response to
the incoming update, peak and recovery should not be collapsed into one timestamp.

### Representation tracing and learning interpretation: C7

[061](../../../analyses/061-2026-11-07-mlp-adam-spike-internals/README.md) and
[062](../../../analyses/062-2026-11-07-mlp-gradient-amplification/README.md)
establish the MLP investigative chain. Hidden weights dominate the static
counterfactual loss decomposition; changing feature direction, reduced
cancellation and output-sensitive motion matter. Batch-gradient growth of
4.13/5.33 times accompanies individual-gradient RMS growth of 1.63/1.88 times,
while the measured Jacobian RMS changes only about 0.5%/0.36%. These results
support coherent amplification beyond a simple uniform Jacobian-size account.
The model has tanh and no RMSNorm; a normalization-denominator-collapse account
does not describe this case. Static decompositions are distinct from intervening
on the evolving trajectory.

[063's method](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md)
standardizes fixed probes, outgoing representation changes, layer/stage
aggregation and temporal SVD. Its [consolidated observations](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/004-consolidated-instability-tracing.md)
and [event-window report](../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md)
cover MLP, ResNet and Transformer. The probes are all 5,000 MLP examples,
1,000 ResNet examples as an intact BatchNorm group, and 64 Transformer sequences
with 2,240 tokens; the latter two are diagnostic subsets of their training
objectives. The selected event windows are MLP 450–500, ResNet 280–310 and
Transformer 552–600, all seed 0.

In the short MLP window, all stages have negative adjacent displacement cosine
in all 51 measured updates. Output top-1/top-10 temporal energy is 53.02%/99.48%,
with rank 5 reaching 95%; the leading coefficient changes sign 42 of 50 times.
ResNet and Transformer output rank-95 values are 16 and 15, respectively.
This is strong evidence of structured, concentrated oscillation, with dimensionality
reported per case. It does not require all architectures to be one-dimensional.
At Transformer update 558, embedding/attention/output cosines
+0.390/−0.672/−0.920 localize a candidate onset boundary. The longer MLP
448–800 raw-gradient/output alignment correlation is 0.776; the short-window
value is 0.642, so these windows must not be mixed. ResNet/Transformer reported
correlations are 0.705/0.879. MLP and ResNet replay checks are exact under their
contracts; the Transformer historical loss discrepancy reaches 0.270%.

### Interventions and comparison: C8

The repository already contains the requested LR, clipping, beta2 and local-cap
sequence. Its positive findings can be presented with the actual comparison
and learning tradeoff next to each one:

| Intervention evidence | Result and interpretation |
|---|---|
| [040 first paired actions](../../../analyses/040-2026-11-03-mlp-step-actions/REPORT.md) | Eight arms from MLP state 657 over 128 updates. Half-LR reduces mean loss 34.05%; adaptive-proposal clipping reduces it 18.06%. These are exact common-state comparisons. |
| [044 multispike schedules](../../../analyses/044-2026-11-04-multispike-interventions/README.md) | Fresh common-worker controls at MLP/ResNet/Transformer forks 512/256/544. Immediate half-LR mean changes −25.04%/+27.20%/−9.66%; cosine half-LR −26.30%/+0.62%/−10.68%. The same action has architecture-dependent effects. |
| [050 two-stage MLP decay](../../../analyses/050-2026-11-04-mlp-two-stage-decay/README.md) | A second decrease reduces the subsequent mean/first spike, with no rise above 10% after state 682 in the retained window. The benefit is tied to this schedule and horizon. |
| [051–054 clipping](../../../analyses/054-2026-11-05-resnet-clip1/README.md) | MLP clip-1 raises mean loss 48.01%; clip-0.5 is better than clip-1 but still +2.08% versus baseline. Under cosine-half-LR, clip-1 is inactive in all 288 updates. ResNet clip-1 lowers mean 28.3% and maximum 52.2%, while late/endpoint loss worsens. |
| [043 Transformer clipping](../../../analyses/043-2026-11-03-spark-fixed-step-interventions/README.md) | Short tests already exist at fork 1024: median clipping activates in 3/64 updates with mean −0.058%; twice-median clipping is inactive. The remaining comparison is clipping matched to the selected tracing/cap fork, not the first Transformer clipping experiment. |
| [059 beta2 reduction](../../../analyses/059-2026-11-07-mlp-beta2-reduction/observations/001-beta2-reduction.md) | From state 512, beta2 0.999 becomes 0.95/0.9/0.5, with LR 0.001 and retained moments/clock. Mean training loss changes +16.78%/+18.16%/+127.02%; endpoint test loss falls. Native bias correction changes the initial effective step. 040's beta2-0.99 arm instead used history-aware correction; the two are distinct completed controls. |
| [064 first-layer freeze](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md) | Output reversals in 450–500 fall from 51 to 4 and maximum displacement RMS falls 87.38%; state-474 loss falls 46.43%. Over 448–800 mean loss rises 22.27%. Freezing the layer freezes 93.6% of parameters, so local stabilization and sustained learning are separate outcomes. |
| [065 distributed local caps](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) | q50 caps reduce full-window mean training loss 22.85%/16.22%/26.00% for MLP/ResNet/Transformer; two additional MLP seeds reduce it 19.43%/32.08%. Native Adam state continues; thresholds use eight untreated calibration updates. Test endpoint changes +2.87%/+8.18%/+100.17% make the held-out tradeoff explicit. |

The [065 local-cap geometry](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/010-local-displacement-cap-geometry.md)
defines a per-unit RMS **local affine** response to each proposed weight-row/bias
increment at frozen inputs. It uses the input second moment including bias; it
does not use the temporal SVD and does not directly bound all propagated motion.
The figure-07 page-2 comparison is a scalar baseline/cap cosine-and-RMS heatmap,
with a common baseline normalization. A temporal spectrum/coefficient display
can adopt its visual organization while retaining this distinction.

The latest [quantile/LR ablation](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md)
is complete: 54 continuations, three initializations, five quantiles crossed
with LR 0.001/0.002/0.004, and an uncapped control at every LR, all from state 448
through 1152. Each test evaluation covers all 10,000 examples; there are 23
sampled evaluations. q50 at 1x gives the lowest observed across-seed mean
training loss (paired reduction 24.79%) and lowers positive loss variation 80.66%,
with final-four test mean +3.37%. Every higher-LR cap improves upon uncapped Adam
at that LR, but every quantile has a worse across-seed mean at 2x/4x than at its
own 1x. The result establishes tolerance of larger nominal LR under a cap,
not accelerated learning. Three seeds give descriptive replication, not an
optimal quantile. Complete retained binaries for this latest sweep remain on
Spark after the local disk limit; compact verified results are local. Planning
must not assume that every full replay is currently possible from local files.

## VI. Conclusion — supported synthesis

The existing evidence supports the intended progression: an update-based
characterization, its curvature/noise/loss connections, controlled GD and SGD
tests, an interpretation along adaptive paths, structured representation-level
oscillations, and interventions that change those dynamics. The conclusion can
replace outdated proposals for tracing or first causal controls with the work
already completed. Generalization, inexpensive deployment, advance prediction
and acceleration remain separate claims with their own evidence requirements;
they need not displace the positive explanatory and training-dynamics results.

## Complete analysis inventory and disposition

Every numbered analysis is represented below. Links point to the owning record;
the table identifies what it contributes and prevents presentation-only or
planned work from being counted as new experimental replication.

| Analyses | Content and present role |
|---|---|
| [001](../../../analyses/001-2026-10-03-model-scale-geometry/README.md), [002](../../../analyses/002-2026-10-04-scale-seed-replication/README.md), [003](../../../analyses/003-2026-10-04-raw-gradient-regimes/README.md) | Pythia size pilot, three-seed replication, raw-gradient regime reanalysis. Motivation and adaptive-scale context for I/V. |
| [004](../../../analyses/004-2026-10-04-lr-schedule-intervention/README.md), [005](../../../analyses/005-2026-10-04-temporal-lag-profiles/README.md), [006](../../../analyses/006-2026-10-07-cumulative-geometry-slopes/README.md) | Schedule intervention; calibrated sketch/exact lag audit; cumulative-slope interpretation. 005 shows negative adjacent and positive all-pair interaction can coexist. 006 shows a B/A plateau can accompany continuing growth and loss reduction. |
| [007](../../../analyses/007-2026-10-08-spark-runpod-timing/README.md) | Operational timing comparison, not scientific confirmation of a regime claim. |
| [008](../../../analyses/008-2026-10-08-no-decay-momentum/README.md), [009](../../../analyses/009-2026-10-09-intermediate-lr/README.md), [015](../../../analyses/015-2026-10-10-spark-high-lr/README.md) | Pythia warmup-only rate/momentum studies, complete three-/seven-rate views and higher rates. Positive moment/adaptive cosine can coexist with poor high-rate loss. Hardware differs across portions of the sweep. |
| [010](../../../analyses/010-2026-10-10-rosenbrock-controlled/README.md), [011](../../../analyses/011-2026-10-10-rosenbrock-lr/README.md), [012](../../../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md), [013](../../../analyses/013-2026-10-10-rosenbrock-adamw-new-start/README.md), [014](../../../analyses/014-2026-10-10-rosenbrock-gradient-noise/README.md) | Exact small-model controls for memory, LR, initial state and injected Gaussian noise. Didactic support for II/III; the injected-noise comparison uses one noise realization and is not a neural seed study. |
| [016](../../../analyses/016-2026-10-11-rosenbrock-hessian/README.md), [017](../../../analyses/017-2026-10-12-rosenbrock-kappa-identity/README.md), [018](../../../analyses/018-2026-10-12-rosenbrock-kappa-noise/README.md) | Exact Hessian, secant identity and noise correction. Distinguish raw projection from spectral curvature and AdamW from SGD. |
| [019](../../../analyses/019-2026-10-13-cifar10-eos-lr/README.md), [020](../../../analyses/020-2026-10-14-eos-multitask/README.md) | Long MLP GD/Adam spectral study; three-architecture fixed-state sampling. Direct foundations for II–IV. |
| [021](../../../analyses/021-2026-10-15-eos-multitask-lr/README.md), [022](../../../analyses/022-2026-10-15-eos-long-horizon-plan/README.md) | Held multitask-LR proposal and prospective long-horizon design. Do not count as completed optimizer trajectories; later calibration work changes their practical status. |
| [023](../../../analyses/023-2026-10-15-eos-smallest-lr-calibration/README.md), [024](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md) | Three smallest-rate calibrations followed by nine completed GD cases. 024 is the active Section IV foundation. |
| [025](../../../analyses/025-2026-10-17-pythia-descent-mechanism/README.md) | Nine 350-update baselines and 18 50-update norm-matched coordinate-scaling branches; fixed disjoint-prefix probe. Controlled support for adaptive descent contribution. |
| [026](../../../analyses/026-2026-10-23-fixed-ce-curvature/README.md) | Ten-seed fixed/native CE arithmetic study on 480 pairs. Strong attained-state curvature difference; matched-loss curvature ratios near one. Supporting context, not proof that mathematically identical loss arithmetic reshapes the exact objective. |
| [027](../../../analyses/027-2026-10-30-minibatch-noise/README.md) | Completed frozen/moving MLP sampling, exact variance checks and independent HVP validation; direct support for C2/C4. |
| [028](../../../analyses/028-2026-10-30-adam-diagnostics/README.md), [029](../../../analyses/029-2026-10-30-pythia-adam-diagnostics/README.md) | Historical Adam synthesis and Pythia scalar-log recurrence/moment diagnostics. Reanalysis of retained evidence. |
| [030](../../../analyses/030-2026-10-30-adam-lr-decisions/README.md), [031](../../../analyses/031-2026-10-31-adam-state-map/README.md) | Eighteen half/same/double-LR branches and exploratory state mapping. Doubling wins final validation in all six parent states despite initial harm. Some early post-action movement features help a small mapping, but no validated pre-action scheduler follows. |
| [032](../../../analyses/032-2026-11-01-eos-exact-lag10/README.md), [033](../../../analyses/033-2026-11-01-eos-long-lags/README.md), [034](../../../analyses/034-2026-11-01-eos-adam-continuations/README.md), [035](../../../analyses/035-2026-11-02-eos-backward-anchors/README.md) | Exact short/long lags, cold-Adam continuations and standardized backward anchors. 034 resets moments and does not match step norm; 035 supplies the active figure. Failed legacy replay profiles are identified rather than silently pooled. |
| [036](../../../analyses/036-2026-11-02-controlled-adam-diagnostics/README.md), [037](../../../analyses/037-2026-11-02-sgd-batch-extension/README.md) | Nine-arm Adam event population and completed three-model/three-seed SGD extension. Primary IV/V evidence. |
| [038](../../../analyses/038-2026-11-03-adam-spike-intervention-design/README.md), [039](../../../analyses/039-2026-11-03-mlp-control-foundation/README.md) | Intervention design/source audit and mock control infrastructure. Neither is a new neural result; later runs execute substantive controls. |
| [040](../../../analyses/040-2026-11-03-mlp-step-actions/README.md), [041](../../../analyses/041-2026-11-03-mlp-temporary-lr/README.md), [042](../../../analyses/042-2026-11-03-mlp-lr-cooldown/README.md) | Paired action screen, temporary LR trigger and cooldown/recovery. Temporary reduction replaces/delays peaks; the near-always-reduced cooldown arm does not establish a selective timing advantage. |
| [043](../../../analyses/043-2026-11-03-spark-fixed-step-interventions/README.md), [044](../../../analyses/044-2026-11-04-multispike-interventions/README.md) | Cross-model fixed-step actions and longer multispike LR schedules, with fresh matched controls where historical replay failed. Includes short Transformer clipping. |
| [045](../../../analyses/045-2026-11-04-mlp-lr-return/README.md), [046](../../../analyses/046-2026-11-04-mlp-cosine-rise70/README.md), [047](../../../analyses/047-2026-11-04-mlp-cosine-floors/README.md) | MLP LR restoration, gradual restoration and decay floors. Abrupt/gradual returns can renew or delay peaks; fewer events need not mean lower mean loss. |
| [048](../../../analyses/048-2026-11-04-mlp-three-paths-to075/README.md), [049](../../../analyses/049-2026-11-04-mlp-direct-abrupt-overlay/README.md) | Presentation of existing runs, not new training. 049 is an explicitly requested intervention-only overlay, supplementary to comparisons with unchanged baseline. |
| [050](../../../analyses/050-2026-11-04-mlp-two-stage-decay/README.md), [051](../../../analyses/051-2026-11-04-mlp-global-clipping/README.md), [052](../../../analyses/052-2026-11-04-mlp-clipping-thresholds/README.md), [053](../../../analyses/053-2026-11-05-mlp-cosine32-clip1/README.md), [054](../../../analyses/054-2026-11-05-resnet-clip1/README.md) | Two-stage schedule, MLP clipping thresholds, combined LR/clipping and ResNet clipping. Preserve activation fractions and unchanged baselines to interpret each complete intervention. |
| [055](../../../analyses/055-2026-11-05-adam-directional-amplification/README.md), [056](../../../analyses/056-2026-11-05-gd-paired-displacement/README.md), [057](../../../analyses/057-2026-11-05-adam-response-memory/README.md) | Population/severity reanalysis, GD/SGD accumulation reanalysis and response-memory checks. Retained evidence is useful despite rollback of their proposed prose/notation. |
| [058](../../../analyses/058-2026-11-06-intervention-consolidation/README.md) | Nine consolidated comparison figures from existing runs. Main source for the author's requested Adam overview/LR/clipping presentation. |
| [059](../../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md), [060](../../../analyses/060-2026-11-07-mlp-adam-to-gd/README.md) | Completed beta2 branches and Adam-to-GD switches. The latter changes optimizer and LR together; GD 0.02 suppresses rises but worsens mean/tail loss over the comparison window. |
| [061](../../../analyses/061-2026-11-07-mlp-adam-spike-internals/README.md), [062](../../../analyses/062-2026-11-07-mlp-gradient-amplification/README.md), [063](../../../analyses/063-2026-11-07-architecture-instability-tracing/README.md) | MLP internal investigation, coherent amplification and unified three-architecture tracing. Actual measurements and formal method support C7. |
| [064](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md), [065](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md) | One continuing intervention study: freezing, LR changes, selective controls, local caps, cross-model comparison, MLP replication and completed quantile/LR factorial. Continue work in 065 under the repository contract. |

## Completed work, remaining links, and workflow

Several old future-work items are now completed and should not be reissued:

- Neural curvature/noise validation exists in 027; three-model batch testing in
  037; exact longer-lag checks in 032–035; all-pair accumulation checks in 024/056.
- Beta2 reduction is in 059, with a distinct history-aware arm in 040. Transformer
  clipping exists in 043 at a different fork. The H049 quantile/LR grid is complete.
- The MLP tracing records' early causal-control TODOs are followed by 064/065.
  Norm-matched scaling, selective layer rates, neuron/local caps, direction removal
  and movement gates have already been tried. The norm-matched MLP control lowers
  mean loss only 3.43% versus 22.85% for the distributed cap; its schedule is
  retrospective. Direction removal/gating failures are retained controls, not
  reasons to erase the positive cap result.
- Reformatting 048/049/058 is not new replication. Reapplying the rolled-back
  C/D or optimizer-state prose is not an authorized manuscript edit.

The immediate remaining work is consolidation: map each equation and result to
the current claims, choose compact existing validation, construct the time-correct
peak table, and prepare the requested tracing/intervention displays. The current
[task assignments](../work/TASKS.md) own those deliverables. A new measurement is
needed only for a specific additional claim: a same-fork clipping/LR comparison,
larger-model seed replication, probe robustness, measured efficiency, or an
advance-warning test with information available before the outgoing update.
These are local evidence gaps, not an imposed new experiment campaign.

In particular, 030/031/055/057 do not validate an advance predictor. In 057,
adding cosine to the log-norm severity family worsens held-out-rate MSE in all
seven rate-held-out folds, and there are no independent seeds. This limits a prediction
claim while preserving the diagnostic and explanatory results. Likewise,
lower training excursions in 065 should be reported alongside the measured
held-out tradeoff; no new generalization claim is needed to describe the
observed representation-control effect.

Any subsequent execution follows the owning [workflow](../../../research/WORKFLOW.md)
and tracing contract: preserve frozen scientific inputs and raw evidence, record
state/update indices and probe/objective coverage, and compare every complete
intervention against its unchanged baseline. Changed executed inputs receive
new run IDs; the ongoing MLP study remains Analysis065. Missing histories and
known replay differences must be declared before promising a reanalysis. This
document authorizes no training, spending, submission or manuscript changes.
