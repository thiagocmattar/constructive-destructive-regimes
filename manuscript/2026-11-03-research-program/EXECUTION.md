# Experiment specifications and execution gates

This is the mutable forward program chosen for design and preparation, following the author's request for a stronger paper. The approved local Run218/Analysis040 eight-arm MLP pilot has now completed in 64.06 seconds with zero cloud spending; see [its empirical report](../../analyses/040-2026-11-03-mlp-step-actions/REPORT.md). The next proposed assignment is the focused E3 visualization pilot below. Broader E1-E4 work and Armijo remain proposed. New scientific inputs require a new run; allocate IDs from the current registries, not the historical pointers in the original preparation notes below.

## E0 Completed mathematical preparation

The NumPy-only `mathematical_preflight.py` checks 140 quadratic cases, 90 smooth quartic cases, 690 direction/path transport bounds, BB equivalence, quadratic line-response identities, explicit endpoint-noise contamination, undefined inputs, and two adversarial controls. Its helper checks the raw scalar formula, positive curvature and an upper cap; it is not the production A8 implementation. Exact tests of A8's additional secant-angle tolerance, lower floor and fallback remain future driver work. The receipt records finite numerical checks, not a proof of a new theorem or a neural-network finding. No HVP workload or model training was run.

One control holds the entire observed preceding secant fixed while unseen current-direction curvature varies by four orders of magnitude. The other holds a period-two loss envelope fixed while an adjacent sign/growth alarm repeatedly fires. The latter still has recovered loss excursions; its purpose is to distinguish targets.

## E1 Fresh geometry and frozen-state action calibration

### Primary comparison

Test whether historical temporal features predict error in applying old curvature to the current adaptive direction, beyond simple current-pair and direction-rotation information. A second target tests whether the same information improves the choice of scalar step on the actual nonlinear loss. The compact strong comparator includes current norm/growth, current-pair statistic, angle to the last step, secant age, conventional positive-curvature/consistency checks, and training age/phase. Give fitted comparators comparable capacity and selection effort.

For controlled discovery, keep the existing three fixed objectives and ordinary Adam recipe, but create new frozen inputs and capture rather than append new science to Run215. Initial exploratory seed0 may qualify historical replay; independent development seeds and test seeds must be separate. Retain a lower, middle and stress LR determined without the new confirmation outcomes. Fresh matched initialization/data/optimizer state is shared across methods within a case/seed.

On the cheap MLP, the proposed development seeds are 0–4; independent confirmation initially uses 5–10. These are new trajectories/instrumentation, although seed0 is also a historical development realization. They do not require a cloud rental. For ResNet/Transformer transfer begin with three new initializations, with matched runtime within each seed block. Three seeds are a transfer check; expand if the primary claim needs tighter independent uncertainty.

Predeclare endpoint windows at states 256, 512 and 768, and diagnostic checkpoints at 128, 256, 384, 512, 640, 768, 896 and 1024, with a 1024-update primary horizon. Use fixed windows rather than selecting only the best-looking spike. Separately mark ordinary constructive, bounded-reversal and growing-loss contexts using a loss/history rule defined on development data. A stratified retrospective context analysis must remain distinct from the unconditional primary checkpoint grid.

### Measurements

- At every state: complete loss, raw norm, adjacent products/cosine, moment/proposal/applied norms, moment ages, LR, and exact availability timestamps.
- Cheap history: lag-two norm growth and pair-sum/pair-difference norms; these need no dense 256-vector online buffer.
- Selected exact history: K=2/4/8/16 current-direction coverage, Gram/step-history rank and conditioning, secant age and consistency. Test incremental value beyond lag-one/current-direction agreement and a flexible recent-norm/loss comparator.
- Selected longer windows: exact anchored lags through 256 in the controlled models, with shared endpoints/horizons and amplitude retained. Current backward-anchor and forward-anchor conventions must not be merged. Larger-model dense histories need an explicit byte budget before capture.
- At fixed calibration states: one independent current-direction HVP and exact gradient/step products; compare endpoint curvature with the completed-step average rather than calling them the same quantity.
- At those states: scalar line-response factors `{0, 0.125, 0.25, 0.5, 1, 2, 4}` times the baseline LR, all from the identical state and staged current Adam direction. Factor0 is a no-movement reference. Also evaluate every eligible continuous method proposal, including values between grid points or below0.125. Compute nonnegative loss regret against the best evaluated candidate in the union of fixed-grid and method-proposed scales, and show the fixed grid separately. This is a finite-candidate reference, not the continuous optimum. Restore weights, moments, clocks, RNG and buffers between probes. If the grid's best endpoint is at its largest factor, record boundary limitation. Charge all added candidate evaluations.

The HVP is a scientific reference, not an input available to a claimed no-extra-backward policy. Report absolute/signed errors, positive-curvature log ratios only where defined, and loss regret relative to the best evaluated candidate defined above. A predicted quadratic optimum and the best observed nonlinear step answer different questions. Report predeclared benign/stress conditions separately as well as pooled; a phase or LR contrast alone is not a within-condition reliability result.

In stochastic AdamW, compare independent sampled endpoint gradients with a fixed/common-probe derivative at selected states, and measure probe cost separately. Same-batch probes do not automatically equal population/full-objective curvature. Do not silently change the sampling unit, normalization, dropout or complete-validation coverage.

### Exit criterion

Freeze one candidate feature/rule only if it improves calibration or action quality consistently beyond current-pair/direction-only controls on development-held seed blocks. Direction coverage is one candidate; temporal coherence is another. Do not choose a gate using the independent confirmation cohort. A history-negative result can still support an observability limitation, but it must not become a positive temporal-controller claim.

## E2 Nine-arm matched action pilot

Retain the eight exact interventions from Analysis038 and add one conservative spectral reference. All nine share the full model/optimizer/RNG state within a fork. The main selected forks are MLP657, ResNet383 and Transformer755; each is 32 updates before its historical formal pre-rise. Continue 128 updates. Fixed rates are 0.001/0.003/0.003 respectively, seed0; no existing weight decay, clipping, warmup or schedule.

| Arm | Action | Interpretation |
| --- | --- | --- |
| A0 | Ordinary Adam unchanged | Contemporary counterfactual |
| A1 | LR x0.5 | Moderate scale reduction |
| A2 | LR x0.25 | Dose and progress tradeoff |
| A3 | Raw full-objective norm clipping | Changes input to both moments |
| A4 | Complete Adam proposal norm clipping | Direct motion cap with ordinary moments |
| A5 | Current raw gradient replaces m-hat in numerator | First-moment ablation; direction and norm both change |
| A6 | beta2=0.99 with historical bias correction | Faster denominator response |
| A7 | Full-objective Armijo, rollback and descent fallback | Expensive accepted-loss safeguard reference |
| A8 | Cap-only transported-secant step | Cheap scalar curvature action, with explicit fallback |

A0–A7 follow the [existing protocol](../../analyses/038-2026-11-03-adam-spike-intervention-design/PROTOCOL.md), including prior-only medians over 64 states, beta2 correction `1 - 0.999^f * 0.99^k`, Armijo coefficient1e-4, halving factor0.5 and at most eight trials. This new nine-arm packet does not edit that historical design in place.

### A8 development specification

At state t, after computing current g and staging the ordinary Adam moments/direction d, use the previous applied increment s and difference y=g-g_previous. Compute `k_hat=(s dot y)/(s dot s)` and `alpha_hat=(g dot d)/(k_hat*(d dot d))`. This references the actual preceding increment, not an LR-free direction mislabeled as displacement.

Use the proposal only for finite inputs, reductions and alpha_hat, nonzero s/d, `g dot d > 0`, and `s dot y > 1e-8*||s||*||y||`. This last condition is a declared development tolerance on secant alignment, not proof that physical curvature is numerically resolved from FP32 gradient differences. When valid set `alpha=min(base_lr,max(0.05*base_lr,alpha_hat))`. Otherwise apply the unchanged baseline LR and record the exact fallback reason. This is a cap-only reference, not a proof of monotone loss or an unconstrained optimal-step method. Both the floor and numerical threshold are development choices to freeze before confirmation. A two-sided adaptation candidate must be a separately named comparison if later needed.

Keep normal Adam moments unmodified by the scalar cap. Record alpha_hat before clipping, chosen alpha, clamp/fallback fractions and actual parameter displacement. At the fork, inherit the preceding gradient and actual step from the newly generated prefix; do not manufacture them from an unavailable historical checkpoint.

### Replay and control requirements

Run215 overwrote intermediate recovery states. Regenerate one common prefix per case; preserve immutable snapshots at fork and formal pre-rise/onset, and a separate rolling crash-recovery file. Preserve full diagnostic/optimizer age, RNG, data position and normalization state. Verify a fresh untreated branch reproduces its common prefix/suffix on the same device. Historical replay qualification uses the explicit Analysis038 timing/amplitude tolerances; if it fails, record that and amend event selection before inspecting treatment outcomes.

Primary selected-event outcomes use the original untreated calendar-state reference and common window: maximum positive excess, excess area, delayed/replacement spikes, final/mean loss and time-to-target. Include censored events and incomplete arms. Charge extra forwards, backwards, state saving and transfers. The existing >=50% excess reduction and <=1% loss penalty is an exploratory screening rule; continuous results remain primary evidence for interpretation.

A selected historical fork only identifies conditional action effect. Add cheap ordinary-state forks on MLP during development to detect over-intervention costs. E3 is required for a diagnostic timing claim.

## E3 Timing and mechanism isolation

Freeze one action selected by E1/E2. Compare identical action magnitude/duration with norm-only, loss-only, current-pair reversal/growth and selected history/validity triggers. Include no intervention, always-on action and a fixed LR comparator. Calibrate gates to comparable intervention budgets on development histories, then freeze thresholds; do not force equal realized counts on test paths after observing outcomes.

For direction-changing treatments, include a native-direction control rescaled to the same treatment step norm; this asks whether the direction adds value beyond scale. For a scalar-step candidate, record and compare cumulative step scale, intervention counts and a prespecified timing placebo if needed. A gain attributable to smaller steps is a scale finding. An expensive HVP or line-search oracle may be used offline to interpret action quality, never hidden in the cheap method's cost.

The raw-gate idea uses current/previous gradients before the outgoing update. Global clipping does not change the current pair's cosine; it changes subsequent motion and, for Adam input clipping, both moment recurrences. Filtering a gradient, skipping an applied step, freezing moments and rollback/retry must be distinct named arms.

### Next proposed test: temporary LR reduction around the MLP spike

Status: design for discussion, not a launch instruction. This is a controlled
full-objective MLP development pilot, reusing the authenticated seed-0 state 657
from Run218. It asks whether temporary diagnostic-conditioned LR reduction can
mitigate the selected excursion while retaining useful training progress and
spending fewer updates at reduced LR than the always-half comparator.

**Manuscript terminology.** Use exactly `\eta\widehat\kappa_t` in prose,
captions and diagnostic axes, as defined in the active editable
[`03-methods.tex`](../draft/sections/03-methods.tex) and
[`06-adam-diagnostics.tex`](../draft/sections/06-adam-diagnostics.tex):

\[
\eta\widehat\kappa_t
=1-\rho_t\frac{\|g_t\|}{\|g_{t-1}\|}.
\]

Keep `\kappa_t` for the directional curvature defined in the methods and
`\rho_t` for consecutive raw-gradient cosine. Use the manuscript's estimated
quantity for the measured diagnostic; do not replace it with a new scalar
name or call it an LR cap. Under Adam it remains the observed projected
raw-gradient change along the realized path. With a varying LR, compute the
same observed gradient-pair expression directly; do not rescale it into a
forecast of curvature or presume that halving LR halves the next diagnostic.
Executed source fields and raw artifacts retain their original names.

**One fixed candidate rule.** At each optimizer boundary, compute the diagnostic
from the current and preceding available raw gradients before applying the
outgoing update. If it is at least 1.8, use LR 0.0005 for that update; otherwise
use the base LR 0.001. Always choose between these two absolute values: no
compounding reductions, permanent first-alert drop, smoothing, persistence,
threshold grid or changed Adam moments. A finite-gradient boundary with an
undefined diagnostic uses base LR and logs the reason; ordinary nonfinite
training failures retain their existing failure handling. The value 1.8 is a
candidate heuristic selected for this development discussion, not an Adam
stability boundary or a validated optimum. Previously inspected untreated
states 657..784 meet the 0.8 threshold at 127/128 boundaries and the 1.8 threshold at 71/128; these are
retrospective control frequencies, not intervention frequencies after feedback.

| Condition | Applied LR |
| --- | --- |
| Unchanged Adam | 0.001 throughout |
| Always-half LR | 0.0005 throughout |
| Temporary diagnostic-conditioned reduction | 0.0005 when the available diagnostic is at least 1.8; otherwise 0.001 |

All three conditions start from the identical complete state 657 and continue
to state 1024: 367 updates each, 1,101 total, 5,505,000 nominal training-image
update exposures. Preserve Run218's 5,000 CIFAR-10 images, tanh MLP, half-squared
error reduction, Adam betas (0.9, 0.999), epsilon 1e-8, no decay or pre-existing
clipping/schedule, and numerical recipe. Reuse the saved parent rather than
retraining its prefix. Boundary/evaluation/qualification work is additional and
must be counted separately. Zero cloud spending; provisional local expectation
1-3 minutes, hard limit 10 minutes, based on the earlier 64.06-second pilot,
not a new measured timing. Neural qualification and production both require
explicit confirmation of this revised plan. Retain the parent, per-update
scalar histories, start/end/recovery state and status; do not add a dense
all-step vector archive just for these plots.

**Figure contract.** Fix the event using the unchanged control: pre-rise 689,
peak 703, recovery 729. Use the same calendar-state zoom 673..745 for all three
conditions, including treatments with no peak or a shifted peak. Three stacked
panels share the state axis: unsmoothed training loss; measured
`\eta\widehat\kappa_t` with the 1.8 threshold; and actual applied LR divided by
base LR, drawn as a step function. Shade the conditioned method's intervention
intervals consistently across panels. Align the diagnostic and outgoing LR to
the same decision boundary; the resulting loss appears after that update.
Include a separate full continuation view 657..1024 to expose delayed/replacement
excursions and LR recovery. Do not recenter each condition on its own peak or
crop the comparison after observing the treatment.

**Readout.** Use the shared reference B = untreated loss at state 689 for peak/area over states 689..1024; report mean and
final training loss over the complete continuation (post-update states 658..1024), later/censored excursions,
reduced-LR update fraction and LR-switch count. Frequent LR alternation or an
almost-always-reduced LR is a result, not permission to tune the rule silently.
Endpoint held-out evaluations may be descriptive but do not select the rule.
A favorable development result would combine lower excursion severity with
competitive progress and fewer reduced-LR updates than always-half. This
single selected event cannot establish robustness, generalization, or a special
advantage of diagnostic timing over an equally frequent alternative trigger.
Independent replication and a timing/budget control would be needed for those
stronger claims. Start with this one MLP example before considering the other
manuscript spike examples.

## E4 Confirmation and practical AdamW

After development, freeze target, diagnostic/window, thresholds, step caps, fallback, method code and comparator tuning budget. Proposed primary inexpensive confirmation: six unseen MLP seeds. Proposed transfer starts with three seeds per selected larger controlled case. The predeclared tests include all trajectories, including no-spike and diverging ones.

For a practical method claim, use a new pinned Pythia/MiniPile AdamW cohort, with random initialization, standard causal token reduction, accumulation, schedule, clipping and decay. Keep data order, tokens and validation coverage matched within seed. The existing 14M/70M recipes are useful starting points, not mandatory constraints on a better clean design. A natural primary is70M; use14M for development and additional cheap replication when useful.

Required comparators depend on the claim: a well-tuned ordinary AdamW recipe, norm/loss controls, and the closest faithful step-adaptation baseline(s), especially GALA and H_ADAM for competitive scalar LR adaptation. Implementing their names with different numerical conventions is insufficient. Charge their extra reference gradients and searches, and compare both equal tokens and measured wall/evaluation budgets. The baseline tuning effort must be comparable to candidate selection. Choose the exact seed count/method roster after calibration and variation estimates; additional credit should buy adequate replication rather than a single larger demonstration.

Do not use confirmation validation/test outcomes to choose the candidate. Freeze a primary performance outcome and noninferiority/effect margin after development, before the test cohort. Report paired seed differences and uncertainty, time-to-target, excursion area/maxima, recovery/censoring, false interventions, costs and memory. Different objectives/units and different fixed normalization groups stay explicit.

## Engineering work before the first neural run

The production implementation is not yet written; the concrete plan/config below separates design readiness from launch readiness.

1. Allocate a new run/analysis under the current registries and pin sources/config/initial tensors/cache hashes. Do not mutate executed Run215 files or mix the old FineWeb harness into this study.
2. Reuse `ProductionTask`/model construction. Add a run-local pre-update observer that sees g_current, g_previous and s_previous. Existing `Diagnostics.consume` is post-update and cannot alone supply the action clock.
3. Implement A0–A8 with ordinary moments, full-accumulation clipping placement, independent beta2 age correction, explicit parameter proposals, FP32 applied-increment audits and complete rollback for A7.
4. Add immutable fork checkpoints plus rolling recovery, status/tracebacks, source identity and scalar/vector audit products. Selected history windows get a byte/retention budget; do not dump every dense vector by default.
5. Focused tests: reference moment recurrences; A0 equivalence; zero/nonfinite cases; no outgoing-data leakage; clip placement; capped proposal; bias correction; RNG/BN/clock rollback; accepted/rejected Armijo trials; exact resume; same fork identity across arms; prior-only thresholds.
6. After the explicit run-plan confirmation: mechanics smoke, production-shaped timing/peak-memory calibration, full selected pilot, local source/numerical checks, figure review, then scientific consolidation.

Tests and synthetic algebra are reversible preparation. A neural mechanics smoke/calibration is training and follows the same run-plan confirmation gate. No smoke was run in this planning turn.

## Concrete first local plan

See [pilot_plan.json](pilot_plan.json). It specifies controlled full-objective CIFAR-10 MLP, Adam(0.9,0.999), seed0, baseLR0.001, fork657, nine128-update branches,64 prior scale states, full5,000-image half-squared-error objective, FP32 model/moments and FP64 products. Nominal objective exposure is9,045,000 training-image updates from1,809 optimizer updates; mechanics/evaluations/rejected forward trials are additional and separately counted.

Armijo may add up to1,024 full-objective forward trials in the MLP branch. Full10,000-image test evaluation at fork/end is descriptive and does not select the method. The first study uses training criteria for development. The initial stochastic/practical confirmation must have an appropriate held-out development/validation/test split and frozen selection procedure.

Provisional expected local mechanics/calibration window is5–15minutes; hard bound30minutes; zero paid rental. Historical ordinary MLP updates were0.01168seconds and an entire1,024-update arm14.90seconds, but new probes, saving and interventions are unmeasured. This is a qualified planning allowance, not a measured production ETC. Once implemented and confirmed, replace it with actual calibration.

## Cloud envelope and cleanup

Available balance at the read-only snapshot isUS$23.6127. The proposed first-stage limit isUS$20 aggregate: at mostUS$2 paid qualification,US$13 selected production andUS$5 retrieval/storage/retries/contingency. Those allocations sum to one ceiling. At most two workers;12-hour task wall deadline; price ceilingUS$1.19/hour unless an explicit amended plan improves cost without exceeding the total. Requery GPU capacity, price and disk charges before creating anything.

Begin retrieval/teardown byUS$16 of accumulated active task compute spend, and earlier when worst-case remaining retrieval/storage would exhaust theUS$20 ceiling. The remaining allowance is a reserve for all task costs, not permission to add production. Maintain a separate accounting ledger for task-owned resources and preserve unrelated resources/spend. An independent guard must actually read/stop/delete its task resources and verify state; an unenforced provider timeout flag is insufficient.

The nine-arm three-model design is5,251 nominal updates including1,795prefix updates. MLP is local. Historical paid4090 medians imply5.99GPU-hours of ordinary work; Armijo pure-forward cost has not been separately calibrated. This design may need staging or narrowing to fit live hardware. Admit only work that fits the remaining measured envelope; record incomplete coverage rather than exceeding it.

Use detached execution, per-update manifests/events, immutable fork states, complete numerical/status records, bounded monitoring and local SHA-256/byte verification before deleting task resources. Report retained storage and provisional/posted cost separately. Refresh the portable CLI and pin its version before provisioning; current authenticated read access is working.
