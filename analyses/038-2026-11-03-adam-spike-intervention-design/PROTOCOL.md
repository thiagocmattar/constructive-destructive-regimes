# Adam spike intervention proposal

The first question is causal and local: from the same weights and Adam state,
which interventions reduce a selected loss excursion without paying for it with
worse subsequent optimization? A second experiment would ask whether an online
rule delivers the same benefit without knowing when a spike will occur.

H028 is proposed, not tested. This packet prepares the design and audits existing
evidence; implementation, training and spending await the author's evaluation.

## Selected events and earlier forks

Use the largest recovered relative excursion at the highest tested LR for each
model, exactly the three events in the requested figure. Preserve the original
loss-only definition and historical evidence. Parameter state t follows t updates;
g_t below is the gradient evaluated at that state, before update t to t+1.

| Case | LR | Formal pre-rise a | Peak p | First recovery c | Peak / pre-rise | Main fork a-32 | Last state after 128 new updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 / MLP | 0.001 | 689 | 703 | 729 | 4.1763 | 657 | 785 |
| CIFAR-10 / ResNet-32 | 0.003 | 415 | 422 | 482 | 9.5877 | 383 | 511 |
| WikiText-2 / Transformer | 0.003 | 787 | 794 | 855 | 3.3140 | 755 | 883 |

The original displayed windows start at 693, 412 and 784. The MLP window already
begins inside its formal excursion; the shaded +10% crossing is not its pre-rise
boundary. Main forks start 32 updates before a, using the 64 strictly earlier
states [a-96,a-33] to set clipping scales. The supplied extended-context CSV
covers [a-96,a+128], 225 observations per model, without smoothing.

Raw evidence sharpens the intervention question. All 28 updates at states a
through p-1 have g_t dot Delta_t < 0 and positive cos(g_t,d_t), where d_t is
Adam's LR-free direction. Loss still rises. A rule that acts only when Adam's
step has a nonnegative gradient dot product would not trigger on these rises.
This is compatible with finite-step overshoot; it does not identify the Hessian
or prove that a particular optimizer component causes the peaks.

Negative consecutive raw-gradient cosine is also common before intervention:
61/64, 26/64 and 15/64 states in the proposed threshold histories. These are
descriptive counts, not a false-positive-rate estimate. A sign-only alarm needs
an ordinary-training control before it can be called useful.

## Matched scientific inputs

Retain the Run215 seed0 frozen initialization, prepared data bytes, source
revisions, model definitions, ordering, parameter scope and physical partition.
The exact identities are pinned by its input manifest and this analysis's
source review. Reverify cache hashes at execution rather than assuming that a
file at the same path is the same input.

| Case | Objective of every full update and safeguard trial | Partition |
| --- | --- | --- |
| MLP | Mean half squared error, summed over 10 outputs, on the same 5,000 prepared CIFAR-10 training images | Fixed chunks of 1,000 |
| ResNet-32 | Mean CE over all 50,000 CIFAR-10 training images | Same 50 fixed ghost-BN groups of 1,000; train-mode statistics without running-buffer mutation |
| Transformer | Token-weighted NLL over all 59,680 prepared causal sequences and 2,088,600 valid targets | Fixed chunks of 64, length 35, padding excluded; dropout zero |

The Transformer vocabulary retains the historical train/valid/test union and
the original tail/stream exclusions disclosed in Analysis036. Changing that
protocol would require a separate comparison. Sample units are images for
MLP/ResNet and prepared sequences with a valid-token denominator for Transformer;
an optimizer boundary consumes the entire fixed objective, not one chunk.

Base optimizer: ordinary Adam, beta1=0.9, beta2=0.999, epsilon=1e-8, no weight
decay, warmup, schedule, AMSGrad or existing clipping. Use FP32 parameters,
gradients and moments; FP64 scalar products and diagnostic reductions; TF32 off,
foreach=False, fused=False. All branches within a case use the same device,
kernel settings and accumulation order. There is no stochastic batch sampling.

The seed0 pilot is conditional on three deliberately selected historical events.
Repeating deterministic forks does not supply independent seeds. It cannot by
itself establish population robustness, predictive scheduling or generalization.

## Reconstruct and qualify the common prefix

Run215's save routine repeatedly replaced recovery.pt, then renamed the final
state to final.pt. The three retained artifact folders contain only final.pt
and audit gradients at 0, 1 and 1024. They do not contain earlier weights and
Adam moments. Scalar histories cannot reconstruct those missing states.

1. Replay one unchanged prefix from the frozen initialization per case. Save
   immutable, complete fork snapshots at f=a-32, a and a+1, including weights,
   all Adam buffers/ages, RNG, data position, prior gradient/diagnostic state,
   cumulative counters and full scientific identity. Save the f snapshot before
   consuming g_f. No optimizer reset is permitted in the main fork comparison.
2. Continue an untreated branch on the same device. Compare its historical
   prefix loss/gradient norms and its event timing/amplitude with Run215. From
   each saved fork, verify equality to the same newly generated untreated suffix
   for loss, parameters and optimizer state. On nondeterministic kernels, first
   qualify deterministic settings; do not attribute replay drift to treatment.
3. For the stronger claim of intervening on the historical event, require the
   untreated event to have pre-rise/peak/recovery within 2 states of the original
   and peak/pre-rise ratio within 5% relative. These are proposed operational
   qualification limits. Report all deviations and the full control curve.
   Within-machine matching remains required even if this historical test passes.
4. If historical qualification fails, retain the failed comparison. A matched
   experiment may still study a newly generated spike, but reselecting its
   event/forks is an explicit amended design before inspecting treatments.

The main calculation needs prefixes through 657/383/755 and their untreated
128-update continuations; snapshots a and a+1 can be saved from those untreated
continuations without an extra prefix. Extend the untreated replay separately
if a later diagnostic is required; it is not free work.

## Eight arms for the first experiment

Each arm starts from the exact same f snapshot and stays active for all 128 new
updates. Thresholds are fixed from that case's newly reconstructed common
pre-fork history, without treatment outcomes or future peak values. The numbers
below from historical evidence are planning values, not thresholds silently
substituted for a new prefix.

Let h_t denote the optimizer input, m and v the stored moments, and
d_t=mhat_t/(sqrt(vhat_t)+epsilon). The ordinary proposal is Delta*=-eta*d_t.

| Arm | Change at the fork | Purpose and strength of claim |
| --- | --- | --- |
| A0 | Continue unchanged Adam | Contemporary untreated control |
| A1 | Hold LR at 0.5 times the original | Simple reduction in step size |
| A2 | Hold LR at 0.25 times the original | Stronger reduction; exposes a spike/progress tradeoff |
| A3 | Global raw-gradient norm clipping at Cg=median of the previous 64 raw norms | Tests limiting the input to both moments |
| A4 | Cap the complete Adam proposal norm at Cd=median of the previous 64 proposal norms | Tests directly limiting parameter motion while retaining ordinary moment updates |
| A5 | Replace the momentum numerator by the current raw gradient: d=g_t/(sqrt(vhat_t)+epsilon) | Tests dependence on first-moment filtering with the same second-moment recurrence |
| A6 | Change beta2 to 0.99 with history-aware bias correction | Tests whether faster second-moment adaptation mitigates the spike |
| A7 | Full-objective Armijo acceptance, with a descent-direction fallback | Blocks accepted increases in the checked training objective; measures the cost in progress and evaluations |

**A3:** h_t=g_t*min(1,Cg/||g_t||). Apply once after the complete objective gradient
is accumulated and before either moment update. Preserve raw g_t for diagnostics.
Set h_t=0 when g_t=0; do not divide by zero. This changes Adam's future moments.
Clipping each physical chunk would be a different experiment.

**A4:** form ordinary Adam moments from raw g_t, then apply
Delta=Delta* min(1,Cd/||Delta*||). Leave moments and their clocks unchanged by
this scale. A zero proposal stays zero. This exactly bounds the ideal proposal
norm, while FP32 applied increments require a separately measured roundoff check.
It does not guarantee loss decrease. Relative to raw-gradient clipping, the
distinction is established in the [Adafactor paper, Sections 5-6](https://proceedings.mlr.press/v80/shazeer18a/shazeer18a.pdf).
This arm is whole-vector clipping on Adam, not a switch to Adafactor.

**A5:** keep consuming the raw gradient into both stored Adam moments, but bypass
mhat in the applied numerator for the whole branch. This defines a precise
ablation without a cold start or a changed bias-correction clock. It is not a
claim that first-moment alignment equals gradient alignment. Both step norm and
direction can change; a later norm-matched ablation would be needed to separate
those effects.

**A6:** retain v_f and all earlier history. After k new gradients,
v_(f+k)=0.99*v_(f+k-1)+0.01*g_(f+k-1)^2 and
vhat_(f+k)=v_(f+k)/(1-0.999^f*0.99^k). Keep m's original beta1 and age.
The product of the actual historical beta2 values is the correction; changing
beta2 in an ordinary optimizer while reusing 1-0.99^(f+k) would be incorrect.
Numerically stable evaluation and an independent recurrence check are required.

Historical scale values for A3/A4 are:

| Case | Cg | Cd | Untreated rise updates exceeding Cg / Cd |
| --- | ---: | ---: | ---: |
| MLP | 0.626501 | 0.0373113 | 6/14 and 5/14 |
| ResNet-32 | 1.30803 | 0.146927 | 5/7 and 4/7 |
| Transformer | 0.301940 | 1.54105 | 7/7 and 3/7 |

Exceedance counts are evaluated on the historical untreated path. They do not
predict when a treated path would clip. If an earlier median is zero, label the
threshold rule unusable for that fork; do not silently add a new scale. A fixed
2-times-median sensitivity is a later, separately counted comparison if needed.

## A7 acceptance rule and its guarantee

Stage the next ordinary Adam moments once using g_t; do not advance them for
each trial. Set P_t=diag(1/(sqrt(vhat_t)+epsilon)). If g_t dot d_t <= 0, replace
d_t by P_t*g_t. This fallback is first-order descending for nonzero g_t with
finite positive P_t. At zero gradient, record a stationary stop.

Start alpha=the original eta and halve it on rejection. For at most eight trials,
evaluate the exact same full training objective at the actual FP32 candidate
parameters and accept only if both

```text
L(theta_candidate) <= L(theta_t) - 1e-4 * alpha * dot(g_t,d_t)
L(theta_candidate) <= L(theta_t)
```

Reject nonfinite candidates. Each trial starts from theta_t, with identical BN
groups, buffers and RNG; there is no optimizer or RNG accumulation across trials.
On acceptance, commit the staged moments and one optimizer clock increment.
On eight rejected trials, restore weights, moments, clocks, buffers and RNG;
mark the arm stalled and stop it. Do not fill the remaining horizon with zero
steps and call that successful training. Log equality caused by floating-point
rounding and stop if candidate parameters no longer change.

With exact arithmetic, a differentiable objective and a descending direction,
Armijo backtracking accepts a sufficiently small step. For an L-smooth objective,
the descent lemma gives

```text
L(theta-alpha*d) <= L(theta) - alpha*g.dot(d) + L*alpha^2*||d||^2/2.
alpha <= 2*(1-c)*g.dot(d)/(L*||d||^2), c=1e-4,
```

as a sufficient Armijo bound when g.dot(d)>0. A known valid curvature bound is
not available here, so the full-objective acceptance test supplies the observable
safeguard. With a finite trial cap, positive progress is not guaranteed. ReLU
nonsmoothness and floating-point evaluation also limit the smooth proof's scope;
the explicit numerical acceptance check still prohibits accepted rises in the
measured objective. See the [UBC descent and Armijo notes](https://friedlander.io/ubc-cpsc-406/slides/descent.html).

This guarantee covers accepted states of the checked full training objective,
not trial points, unseen data, generalization or convergence of this modified
optimizer. Peak elimination by A7 is a consequence of its rule; the scientific
comparison is useful loss reduction per update, objective pass and wall time.

## Outcomes and measurements

Use the contemporary untreated branch for all paired comparisons. Define
B=L_A0(a), with a held at its selected calendar state. The primary target window
is [a,a+96], which fits all main continuations and covers the historical recoveries.
Report (i) maximum positive excess max(0,L_b(t)/B-1) and (ii) summed positive
excess over B in this identical window, both relative to A0. This keeps an
intervention from changing its own reference to conceal an excursion. Include
absolute losses and absolute differences; report undefined ratios when a control
excess is zero. If an arm stops, report its incomplete coverage and reason;
never compute a full-horizon success score from its shorter favorable prefix.

Also apply the unchanged Analysis036 event detector to each complete branch
joined to its shared prefix. Report maximum observed excursion ratio, counts
above 10%, durations and unrecovered events. Keep an event already active at f
identified as crossing the fork. Include right-censored observed maxima rather
than restricting comparisons to recovered spikes. Events are not required to
retain their original identities after treatment. Search the full 128-update
continuation for delayed or replacement spikes.

Measure useful progress by final loss and mean loss over all 128 post-fork
states, and first attainment of the fixed target 0.95*B (right-censored if
unattained). Report cumulative examples/targets, complete training forward and
backward passes, trial/rejection counts, checkpoint overhead, peak memory and
wall time. Equal accepted updates are the primary matching rule. A7's extra
forward passes must also be charged in objective-pass and time comparisons.
Use time-to-target and common observed compute checkpoints; a claim about loss
at a larger equal-compute budget needs explicitly budgeted longer A0/A1 runs.

An operational promising result requires at least 50% reductions in both target
maximum excess and excess area, final and mean training loss no more than 1%
above A0, complete coverage and no nonfinite/stalled outcome. Show the continuous
scores and all arms even when these thresholds are missed. A7 must additionally
show its evaluation/time cost; suppressing every update is not a useful solution.
These criteria rank a pilot, not establish a statistically validated winner.

Capture every boundary's raw loss, ||g||, adjacent raw cosine and projected raw
gradient change; record clipped input h separately. Capture ||m||, ||mhat||,
||d||, proposal norm, applied increment norm, cos(g,mhat), cos(g,d),
g dot applied increment, clipping factors, effective LR and second-moment age.
Use all unique trainable parameters in fixed order and FP64 products. Zero-norm
cosines/ratios are null with reasons, never epsilon replacements. The
projected-change expression is q_t=1-dot(g_(t-1),g_t)/||g_(t-1)||^2; it remains
a diagnostic under Adam rather than an identified curvature or stability bound.

Record actual-update prefix norm, path length and their ratio from the fork,
with R_k=||theta_(f+k)-theta_f||/sum_(j=0)^(k-1)||Delta_(f+j)||; do not mix this reset
quantity with Run215's initialization-based retention. Save complete recovery
state at the fork, every 32 continuation updates, first rejected/troubled step,
and endpoint. Keep the fork, intervention state, endpoint and audit vectors
immutable; an additional rolling crash-recovery file may be overwritten.

Evaluate every prepared held-out split at the fork and endpoint with complete
coverage and original image/token weighting. Report counts and identities.
CIFAR-10 has the original 10,000-image test split and no new validation split;
WikiText uses all originally prepared valid/test targets. Neither held-out split
selects the method or thresholds; keep these pilot evaluations descriptive and
defer final test interpretation to a frozen confirmation design.

## Later timing and online experiments

After the main pilot, select at most two practical candidates by the fixed
training-loss criteria; report selection as exploratory. Compare each with A0
at a and a+1, 128 updates per branch. This adds at most 18 continuations / 2,304
updates. Recompute threshold medians from each fork's strictly earlier 64 states.
The a fork is retrospective prevention; a+1 is mitigation after the first rise
is already observed. Neither is a forecasting evaluation.

Then freeze one practical policy and A7 as a safeguard reference before testing
new initializations 1, 2 and 3 at the same three model/LR conditions. Start at
initialization and run 1,024 full updates with all event times eligible; do not
select only histories that develop a large spike. Compare four policies: A0,
constant half LR, the selected practical policy, and A7: 36 complete trajectories
/ 36,864 updates before any additional equal-compute controls. The original
three seed0 spikes are development examples, not held-out evidence.

For a selected clipping policy, its deployment form uses a trailing 64-update
scale from its own previously completed history, activated at state 64; the
first 64 updates match A0. This differs from the fixed-scale fork experiment
and must be labeled. A selected changed-numerator/beta2 policy starts at state
64. Include a small timing-neutral comparison before claiming local results
transfer to these deployment forms.

If the paper asks whether its gradient diagnostic adds scheduling value, add a
separate trigger comparison with exactly the same action: hold LR at half after
the first trigger, earliest state 64. Proposed frozen candidates are (i) a
loss-only alarm at >1% above the minimum of the preceding 16 losses, (ii) two
consecutive q_t>2 observations, and (iii) two consecutive negative raw cosines.
Define an observation at t before the outgoing update; require two valid
observations after state 64 and never use outgoing products as predictors.
Evaluate full-trajectory loss, intervention frequency and computational cost,
including trajectories with no spike. These additional trigger arms are not
included in the 36-trajectory count and need their own budget. A temporal
association in seed0 is not sufficient reason to deploy the gradient alarm.

## Additional methods worth retaining as alternatives

- **Secant-based step ceiling:** with s=theta_t-theta_(t-1) and y=g_t-g_(t-1),
  set Lhat=2*||y||/||s|| and
  alpha=min(eta, g_t.dot(d)/(Lhat*||d||^2)) when all denominators and g_t.dot(d)
  are positive. Use the A7 direction fallback for a nonpositive product; when
  ||s||, ||y|| or ||d|| is zero, flag the estimate unavailable and use a separately
  declared fallback. This uses an actual-step secant, avoiding an Adam curvature
  claim for q_t/eta. It is a heuristic: Lhat need not upper-bound curvature along
  the next step, so spike prevention is not proved. It is a useful low-cost
  candidate if A7 works but full-objective trials cost too much.
- **Coordinate denominator floor:** use sqrt(max(vhat_t,g_t^2))+epsilon in the
  denominator with the original mhat. This selectively suppresses coordinates
  whose current gradient exceeds its historical RMS. It is a new ablation here,
  with no loss-decrease guarantee; it can suppress useful coordinate motion.
- **AMSGrad:** test a nondecreasing second-moment envelope as a separate
  alternative. Its [convergence analysis](https://arxiv.org/abs/1904.09237) does
  not guarantee monotone loss or prevent these neural-network spikes. A switch
  at f must declare an envelope initialized to v_f; a genuine from-initialization
  AMSGrad arm requires its own past maxima and is a different comparison.
- **Descent repair or momentum restart on conflict:** cheap to test if new
  spikes exhibit uphill applied directions. It would not activate on the 28
  historical rise updates under the sign-only condition, which limits its
  priority for these examples.

Noise injection, data mixing, changed BN groups and SAM would change sampling,
the objective or its evaluation geometry. They could answer follow-up questions
but would make this first optimizer intervention comparison harder to interpret.

## Work inventory and execution preparation

The recommended first stage is 3 cases x 8 arms x 128 updates = 3,072 new branch
updates, plus 1,795 prefix updates: 4,867 nominal optimizer updates in total.
There are at most 3,072 extra full-training-objective trial forward passes
(3 safeguard arms x 128 updates x 8 trials), plus full held-out evaluations,
checkpointing and qualification. Early stops reduce delivered coverage and are
reported as such; retries and qualification repeats must be counted separately.

Current complete recovery files occupy about 55 MB / 39 MB / 1.17 GB for MLP,
ResNet and Transformer. As a conservative storage scenario, three common fork
snapshots plus four scheduled snapshots per arm would retain 35 files per case,
about 44.1 GB in total at those existing envelope sizes, before failure audits
and input caches. A leaner recovery format may reduce this substantially; measure
its actual size and set a disk/retention ceiling in the launch packet.

Run a mechanics-only local check of moment recurrence, clipping placement,
applied-step rounding, candidate rollback, rejection limits and reproducible
resume before any production. Calibrate complete updates, complete training
forwards, held-out evaluations and checkpoint transfers on the chosen device.
Historical Run215 timings are evidence about its workload/devices, not current
ETC or a new price quote. The first MLP comparison can use owned local hardware;
ResNet/Transformer placement follows the measured cost and available budget.

After the design is selected, allocate new run IDs beginning at the then-current
next available ID (currently 217), freeze the exact inputs and prepare the
implementation and bounded launch packet. A billable launch packet must state
live price/capacity, measured or qualified ETC, worker count, maximum aggregate
cost/duration including retries, disk use, monitoring and teardown. This design
does not provision resources or reuse a historical spending envelope.

Production jobs must survive SSH loss, emit progress/manifests, and be monitored
read-only about every five minutes. Report accepted/attempted updates, tokens or
images, loss, trials, throughput, refreshed ETC and memory; flag nonfinite
values, stalls, replay mismatch and inadequate retrieval margin. Retrieve and
verify file sizes/SHA-256 before removing task-owned compute/storage. Preserve
unrelated queues/resources. Keep weights/caches/logs outside Git, retain raw
failed outcomes, and document any retained storage and provisional billing.

For review, the main scientific choices are the eight-arm pilot, the 32-update
lead and 128-update horizon, and whether the later online confirmation is worth
its larger budget. The recommended first decision is the local matched-fork
pilot; no online efficacy claim is needed to interpret its causal contrast.
