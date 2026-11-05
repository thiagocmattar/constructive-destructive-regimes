# Independent reviewer: validity, replication and experimental return

5 November 2026. This assessment treats the current 18-page manuscript as a
diagnostic paper. It recommends tests, not another rewrite, optimizer or spike
predictor. No experiments were launched, new scientific results computed, or
spending authorized. I reviewed the manuscript and prior evidence independently
of the other new reviewer reports.

## Recommendation

**First recover the independent-seed support already present in the retained GD
reference trajectories. Then, if new training is desired, test raw clipping
versus actual Adam motion at fixed calendar checkpoints across new initializations.**
This best serves the diagnostic utility argument. A mutually exclusive,
eight-trajectory Adam replication alternative is appropriate if the principal
priority is the generality of the rise-phase anatomy claim. Do not bundle both
into an expanding campaign. Both outrank a broad all-rate HVP program for the
current paper.

The main acceptance risk is no longer incorrect algebra. It is whether the
diagnostic account produces a reproducible scientific distinction beyond
identities and selected counterexamples. More dependent events do not address
that risk. A small genuinely held-out state/seed comparison can.

## What is already replicated, and what is not

| Evidence | Actual independent coverage | Consequence for the next experiment |
| --- | --- | --- |
| Main GD learning-rate sweep, Analysis024/056 | Three rates per model; one initialization/model; 3,000 updates | The full rate-dependent pattern lacks initialization replication. Nine trajectories are not nine independent seed blocks. |
| GD references behind Analyses027/037 | **Three initializations in each of three cases, 2,000 full-GD updates each**, at MLP LR .02, ResNet .1, Transformer .2; matched 200-update full-GD continuations from states 0/2000; MLP also state1000 | These already provide an unexploited replication check of the paired geometry at one fixed rate. Do not purchase this coverage again. They do not replicate the complete three-rate, 3,000-update design. |
| SGD comparisons, Analyses027/037 | Three initialization blocks; eight stochastic repetitions/cell; 546 displayed continuations | Repetitions estimate sampling variation conditional on the shared initialization/state. The three blocks, not the 546 paths, support initialization robustness. |
| MLP curvature/noise validation, Analysis027 | Three initializations, all-objective gradients; 1,188 state-matched HVP pairs overall; batch32 contrast uses 288 pairs | Independent HVP calibration is already available for the SGD noise correction. Another identical MLP SGD HVP study has little return. |
| Adam anatomy, Analyses036/057 | Nine trajectories but one initialization/model; 129 recovered >10% events, 127 defined pre-rise cosines; 84 of these are the middle/high MLP arms | Events, rates and updates cannot substitute for independent seeds. Severity fits share both data selection and parent histories. |
| Pythia complete passes, Analysis002/003 | Three paired initialization/data-order blocks at each of three sizes | Already replicated; initialization and data order vary together. More small-Pythia negative-cosine curves are low return. |
| Pythia coordinate-scaling study, Analysis025 | Nine parents, 18 sustained branches, three paired blocks; 81 baseline states and seven one-step candidates/state | Norm-matched component removal, half/double steps and same-state direction controls already exist. Two forks from one parent do not produce six independent blocks. |
| Adam LR/clipping interventions, Analyses040--054 | Many branches, largely one initialization/model and development-selected states/windows | More schedules or thresholds on the same parent mainly expand an exploratory search. One prospective replication of a frozen comparison is more informative. |

Sources: [Analysis027](../../../../analyses/027-2026-10-30-minibatch-noise/REPORT.md),
[Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md),
[Analysis025](../../../../analyses/025-2026-10-17-pythia-descent-mechanism/REPORT.md),
[Analysis043](../../../../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md),
[Analysis057](../../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md).
The held Analysis021 queue is not completed replication and must not be counted.

I inspected the retained Run214 reference scalar schema: every update records
norm, cosine, cumulative A/B/energy, all-pair and lag-one interaction, and the
parallel actual-increment quantities. Its existing verifier audits those fields.
Run186 retains all reference scalar histories but not dense per-step gradient
vectors. Run216 retains the corresponding fresh Transformer reference histories.

## Statements most exposed to reviewer challenge

1. **The C/D factorization adds a useful empirical distinction.** The identity
   R=CD cannot fail except through implementation error. The substantive claim
   is that its components expose reproducible differences hidden by saturated
   adjacent cosines. The highlighted cross-model ordering currently has one
   initialization and different objectives. Confirming an identity on more
   trajectories is less valuable than checking this particular distinction.
2. **All-pair interaction is positive despite local reversal.** This is valid
   for the reported full prefixes. Positive early interaction can dominate a
   long prefix, so the result need not characterize every late local window.
   Prefix length, starting point and pairing offset are scientific choices.
   Analysis056 already checks both pair offsets; proposing that check alone as
   a new contribution would be redundant.
3. **117/127 pre-rise cosines are negative and all peak norms grow.** The
   population is conditioned on recovery, amplitude and loss boundaries;
   reversal can be common outside these events. Analysis057 already checks
   5/10/20% thresholds and censoring, and Analysis055 supplies ordinary-context
   controls. Their remaining gap is duration-matched descriptive comparison,
   not another threshold sweep or forecasting exercise.
4. **794/794 rise-interval steps descend to first order.** This is a useful
   exclusion of the locally-uphill explanation. Its 794 observations are
   dependent and do not identify why the finite step raises loss. The exact
   residual `observed loss change minus linear term` is accounting, not an
   independently measured curvature mechanism.
5. **Smoothing/clipping may worsen loss.** Counterexamples already establish
   that the implication is not universal. The remaining research question is
   how often the diagnostic distinction matters in prespecified settings.
   Single endpoint, maximum and window-mean outcomes can legitimately disagree;
   choosing whichever favors a treatment after inspection would be invalid.

## Ranked tests

### 1. Replicate the central paired-geometry result using retained references

**Decision value: high per unit effort; no new training.** This is the first
test I would request before judging the remaining seed gap.

- **Independent unit:** initialization block within model; report all three
  blocks separately. Runtime differences in the ResNet references remain
  qualified. This is not a pure variance estimate over initializations alone.
- **Grid:** the nine existing 2,000-update GD references; fixed prefixes
  T={200,500,1000,2000}; plus the nine existing 200-update full-GD continuations
  beginning at state2000. Use the common grid without selecting high-reversal
  intervals. Both adjacent pairing offsets cover each complete analysis segment.
- **Matched factors:** existing fixed data/objective, initialization within any
  paired comparison, constant LR, parameter scope, accumulation and precision.
  Label this as one rate/model, not a replication of the complete LR sweep.
- **Primary outcomes:** C, D, R, Q, summed adjacent interaction and loss change
  at each fixed endpoint; actual-increment versus nominal-gradient agreement.
  Ask whether substantial within-pair cancellation and coherent residual motion
  coexist across blocks, and whether negative adjacent/positive total interaction
  occurs on the late independently reset segments as well as full prefixes.
- **Baselines:** adjacent cosine alone, R alone, and the existing zero-cross-term
  reference. Do not fit R from C/D and call that predictive validation: R=CD.
  Show cases with similar cosine but materially different component profiles,
  with all cases retained, rather than selecting just a favorable pair.
- **Falsifiers:** component ordering is unstable across blocks; the central
  distinction disappears outside a full prefix; or the result is dominated by
  one seed/offset. Those findings would bound the diagnostic claim, not invalidate
  the algebra. A missing prechosen phenomenon is retained as a negative result.
- **Uncertainty:** individual block values, ranges and within-block paired
  contrasts. Three seeds do not justify precise population intervals or
  bootstrap inference that treats time points as independent.
- **Artifacts:** source inventories/hashes, per-segment metric table, independent
  reductions of norm/cosine-derived pair norms, endpoint vector audits where
  available, and one compact all-block figure.

**Data limitation:** arbitrary window B is not the difference of two prefix
norms. Restrict this zero-training test to whole recorded prefixes and already
reset continuations. Additional windows require retained parameter endpoints
or cumulative vectors and an explicit numerical audit; dense histories cannot
be presumed to exist. The trained-state continuations already solve the most
important late-window concern without this reconstruction.

### 2. One bounded new Adam study: choose utility or anatomy replication

#### 2A. Recommended: prospective raw-clipping comparison

**Decision value: highest new-work return for the current diagnostic framing.**
Test whether clipping the raw input controls actual motion and whether the
answer depends on the declared outcome/horizon. This is an evaluation of an
existing intervention, not a new clipping method.

- **Minimum grid:** MLP and ResNet, three initialization blocks, original high
  Adam rates .001/.003, parent horizon1024; forks at states256/512/768; unchanged
  versus global raw-gradient norm clipping1 for128 updates each. Six parents,
  18 matched comparisons,36 fresh branches: **6,144 parent +4,608 branch =10,752
  nominal updates**. Four parent/model-seed conditions are genuinely new if
  seed0 is rerun and seeds1/2 are new. Seed0 reruns are controls, not additional
  independent initializations. The128-update horizon is a new prospective
  design, not literal repetition of the old288/320-update selected windows.
- **Unit/matching:** n=3 initialization blocks/model; checkpoints are repeated
  observations within each block. Match exact warm parameters,m/v,time counters,
  RNG,data/objective,precision,physical accumulation/BN grouping and hardware.
  Only clipping varies. No threshold tuning or result-based checkpoint choice.
- **Primary outcomes:** paired128-update mean full-objective loss difference,
  mean and maximum actual-increment norm ratios, and clipping-active fraction.
  Retain separate raw-gradient, clipped-input, moment and actual-increment
  measurements. Endpoint and last32-update mean losses are fixed secondary
  outcomes, not replacements for an unfavorable primary result.
- **Baseline:** fresh unchanged branch at each fork. The original parent segment
  can replace a redundant control only after exact within-runtime replay
  qualification; budget conservatively for all36 branches until that qualifies.
- **Falsifiers:** if the selected MLP increase in applied motion disappears in
  every new block/state, its prevalence is not supported. A loss benefit at
  every fixed horizon would narrow the prior endpoint/window caution to the
  original selected cases. Inactive clipping remains a valid null treatment,
  never a reason to change threshold or move the anchor. Nonfinite cases remain
  in coverage and are not silently replaced.
- **Uncertainty:** all seed-by-anchor paired effects; equal-weight anchor means
  within seed, then three seed summaries and ranges. Do not count18 comparisons
  as18 independent replications or pool the two models as exchangeable units.
- **Artifacts:** frozen protocol/source/config/data and initialization hashes;
  all per-step streams, loss denominators and coverage; immutable parents at
  256/512/768/1024 with full optimizer/RNG/diagnostic recovery; replay controls;
  verified raw histories and terminal states. The high-rate parent histories
  also supply limited new Adam anatomy evidence, but do not replicate the
  low/middle rates or all three model families.

This combines independent seeds with unselected state coverage. Repeating only
the original handpicked fork would leave the state-selection concern open.
No fresh execution/cost claim is made; this is a bounded scientific grid.

#### 2B. Alternative, not additive: eight Adam anatomy trajectories

Choose this **instead of2A** if the author prioritizes replication of the
794/794 rise-phase result over diagnostic intervention utility. It closes a
missing replication dimension without recreating the entire rate sweep or
promising any event recurrence.

- **Minimum grid:** two new initialization seeds, two original rates (low/high),
  and two cases (smooth tanh MLP / WikiText Transformer), each for1,024 updates.
  This is **2 x 2 x 2 = eight new trajectories / 8,192 updates**. MLP rates
  {.0001,.001}; Transformer rates {.0003,.003}. Generate/hash each initialization
  once and share it between the two rates. Use distinct new seeds from all
  exploratory Adam development; literal seed IDs are finalized in the protocol.
- **Why these cases:** MLP supplies a smooth objective relevant to the stated
  curvature assumptions; Transformer supplies a distinct task/modality and the
  existing trained-state alignment/progress dissociation. Low/high original
  rates bracket different observed dynamics without searching rates in the
  new seeds. This deliberately prioritizes coverage of two cases over shallow
  addition of all middle rates. ResNet's Adam claims remain conditional on one
  initialization; this design must not be described as replicating all three
  model families. If the scientific priority is clipping heterogeneity instead,
  replace Transformer with ResNet before launch, rather than silently enlarging
  the grid afterward.
- **Independent unit:** initialization within model. Original seed0 plus two
  new blocks yields n=3/model, with two rates as paired repeated conditions.
  The original seed0 is historical evidence, never a newly held-out test block.
  Check its input/runtime/hardware identities first. If runtime differences
  cannot be bounded, report old/new strata and n=2 new blocks; do not hide the
  confound or automatically purchase reruns. Any requested standardized seed0
  reruns would be four extra trajectories, not independent replication.
- **Matched factors:** fixed complete objectives/data and original initialization
  recipe, physical accumulation, precision, Adam beta/epsilon and horizon;
  no added clipping, decay, schedule, augmentation or dropout. Same runtime and
  hardware within each model's new cohort. In the Transformer, preserve the
  full supervised-token denominator and declared objective coverage.
- **Unselected coverage:** all 1,024 outgoing updates and1,025 losses; fixed
  calendar summaries 0..255,256..511,512..767,768..1023, plus the full history.
  No start point depends on a newly observed excursion. Freeze5/10/20% loss-only
  recovered/censored segmentation unchanged; zero qualifying events remain a
  valid outcome. Do not increase rates or extend a quiet arm to obtain a spike.
- **Primary outcomes:** run-level negative raw-cosine prevalence and distributions,
  raw/moment/actual-update stream distinctions, actual g-dot-increment signs,
  full-history loss change, fixed-window mean loss, and event occupancy/counts
  when present. Among rise intervals, report the descending-step fraction both
  step-weighted and event-weighted, separately by run. Compare pre-rise reversal
  and norm amplification against each run's ordinary background without fitting
  a classifier. Ratios, durations and censoring retain undefined cases.
- **Baselines:** paired low-rate arm, ordinary states within each run, and the
  original independent computation of actual parameter subtraction. Current
  raw-gradient cosine alone is the reduced diagnostic baseline; it must not be
  treated as the actual Adam-direction cosine. A zero-event low-rate arm tests
  ordinary geometry, not failed event collection.
- **Falsifiers:** substantial uphill applied steps in new rise intervals refute
  the extension of the794/794 observation; reversal/norm signatures indistinct
  from ordinary background limit event specificity; large block differences
  limit generality. Quiet new high-rate arms reduce support for recurrence but
  remain scientifically useful. None should trigger a replacement-seed search.
- **Uncertainty:** all per-seed/per-rate values and paired rate contrasts, with
  means and ranges. Summaries first aggregate within seed. Do not treat events,
  time points or rates as independent n, and do not offer precise population
  intervals from two new blocks. No severity regression is needed.
- **Artifacts:** frozen protocol/config/source/data/initialization hashes;
  per-update raw gradient, moment/adaptive and actual-increment reductions;
  fixed-calendar recovery states (at least256/512/768/1024) containing optimizer,
  RNG and prior-gradient/diagnostic accumulators; raw full-objective losses,
  zero-denominator reasons, coverage and exact same-runtime replay qualification.
  These snapshots enable a later small held-state test without rerunning parents.

This improves the replicated existence and scope of Adam distinctions, not the
selection robustness of the old clipping/scheduling interventions. I would not
claim that eight parents alone validate a beneficial intervention. My separate
utility preference is to confirm one frozen raw-clipping comparison at systematic
checkpoints, with actual motion and fixed-horizon loss as outcomes; that would
be a later alternative to a broad HVP campaign, not an automatic extension here.
No current dollar or timing quote is made: throughput and live pricing need
checking if the author chooses execution.

### 3. Calibrate the Adam response approximation on systematic held states

**Decision value: moderate; potentially high only if the memory bridge becomes
a central explanatory claim.** A single smooth-model pilot can determine whether
a larger HVP campaign earns its place. It is not a prerequisite for the bounded
current empirical argument.

- **Grid/unit:** reuse the three fresh smooth-MLP parents from recommended test2A at
  fixed states256/512/768. At each anchor hold the native Adam direction and
  updated P/m fixed. Evaluate one HVP H*g and endpoint gradients at trial scales
  alpha={1,1/2,1/4}. Nine HVP anchors and27 trial endpoints suffice for a
  first comparison; the three initializations, not trial scales, are independent
  units. If alternative2B is chosen, only two new high-rate MLP parents supply
  these snapshots: use six anchors/18 endpoints and report n=2 new blocks.
  If the necessary warm snapshots are absent, prepare a new frozen short cohort;
  do not assert historical event replay. An exploratory seed0 snapshot can
  establish mechanics but cannot count as held-out confirmation.
- **Baselines:** the one-HVP frozen proxy lambda_B, lambda_B plus the measured
  memory correction M, and the actual mixed-direction starting-Hessian
  expression. The latter equals lambda_B+M algebraically and is an
  implementation cross-check, **not a third independent hypothesis**.
- **Outcomes:** signed and scale-normalized errors versus observed z/(alpha*eta),
  correction size and change in remaining error as alpha decreases. Use
  predeclared denominator floors/undefined cases and retain small/zero signals.
  Compare corrected and uncorrected errors within each anchor before seed
  aggregation. No event detector is required.
- **Falsifiers:** correction is negligible or often increases absolute error;
  the remaining discrepancy does not diminish under shorter trial steps beyond
  numerical tolerance; or numerical error prevents separating terms. Then the
  bridge remains an identity/limitation, not an empirical spike explanation.
- **Why smooth first:** the tanh MLP supplies the stated smoothness conditions.
  ResNet and the retained Transformer use ReLU; there the finite-step mismatch
  may include activation-boundary effects. A broad three-model HVP campaign
  without resolving that distinction risks spending money on an ambiguous
  residual. No full Hessian spectrum is needed.
- **Artifacts:** immutable full state, full-objective HVP and gradient identity,
  trial direction/norm, clocks before/after moment recursion, precision audit,
  raw responses and losses, explicit restoration of native parameters after
  every trial. Tests against known quadratics verify implementation only.

Existing Analysis027 already supplies the complementary independent SGD HVP
validation, and Analysis025 already supplies Pythia same-state direction/length
controls. Neither must be rerun to make this narrow Adam comparison possible.

### 4. Finish the conditional-event denominator audit from retained histories

**Decision value: low to moderate, very low compute cost.** It makes the Adam
anatomy defensible, but should not grow into another prediction study.

- **Grid:** all nine existing 1,024-update histories; the frozen 5/10/20%
  recovered/censored incidence already computed in Analysis057 remains the
  starting point. Do not rerun or retune segmentation.
- **New comparison:** compare event rise intervals with ordinary intervals of
  exactly the same duration, same model/rate, start within128 updates, baseline
  loss within factor1.25, and no >10% event overlap. Freeze deterministic nearest
  stage/loss matching without replacement; publish unmatched counts. Also show
  unconditional full-history prevalence so sparse matching cannot hide support.
- **Outcomes/baselines:** pre-interval reversal prevalence, norm-ratio profile,
  actual-step first-order sign and loss excess. These are descriptive matched
  contrasts, not prospective scores. The loss-defined outcome must not be used
  to claim independently predicted loss rises. Analysis055's eight-update
  marker is relevant context but does not supply duration matching.
- **Falsifier:** event signatures match ordinary dynamics or depend on a small
  common-support subset. Then they describe event anatomy without event
  specificity. This still supports the paper's rejection of universal alarms.
- **Uncertainty/artifacts:** arm-level counts/contrasts, matching diagnostics,
  all unmatched events, full source hashes and deterministic matching code.
  There is still one initialization/model; no event bootstrap can manufacture
  independent-initialization evidence.

## What I would not fund first

- The full 27-trajectory, all-rate Adam/HVP program merely to remove every
  qualifier. Its expanded scope is not necessary for the revised argument.
- More MLP LR-return/cosine/threshold variants on the same parent, or a severity
  model with more predictors. Development diversity does not close replication.
- A larger language model just to reproduce negative cosine. Existing Pythia
  experiments already provide replicated language-model evidence.
- Orthogonalized steps, activation sparsity or a new optimizer. They introduce
  a separate methods question and a new comparison/tuning burden.
- Tests that treat all-pair identities, loss residual definitions or R=CD as
  independent explanatory successes. Their role is exact accounting; the new
  evidence must test a substantive conditional pattern or approximation.

## Stopping rule

Run the retained-data test first. If its distinctions replicate across existing
blocks and late segments, the GD seed criticism is materially smaller without
new training. For new work, choose2A or2B, freeze its
grid and complete every admitted condition. Report favorable and unfavorable
results, including inactive treatments and absent events. Add the smooth HVP pilot only to support a desired
mechanistic claim. A null result should close a claim, not reopen an indefinite
search until simulated reviewers prefer the paper.
