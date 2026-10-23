# Prospective comparison and decision rules

## Hypothesis and interpretation

H022: replacing native hard-target cross-entropy arithmetic with the paper's
fixed arithmetic leads AdamW to parameter states with lower loss curvature.
Here, "flatter" primarily means a smaller largest algebraic eigenvalue of the
same empirical training-loss Hessian, in the same parameter coordinates.
Directional and optimizer-scaled curvature answer related, separate questions.

The hard-target loss is unchanged in exact arithmetic. For logits z and target
y, define m=z_y-logsumexp(z_j,j!=y); fixed CE is softplus(-m). At a fixed
theta both formulas therefore have the same exact parameter Hessian. The
logit Hessian is diag(p)-p p^T; the parameter Hessian also includes the
logit-second-derivative term. Neither removal of numerical collapse nor
smoother Adam dynamics implies a monotone reduction in parameter curvature.

This tests the hard-target CE fix. Attention softmax stays native and identical
in all training arms. Distillation and the paper's separate soft-target
cancellation fix are outside the comparison. Results must name this scope.

## Task, model and matched factors

Use (a+b) mod 31: all 961 ordered input pairs. A fixed seed 314159 permutation
defines480 training pairs and 481 held-out pairs; the split is shared across all
initializations and never selected using results. The unequal halves are
declared. One sample is one pair with one target, and the loss is the mean over
all 480 training samples. No language tokenizer, corpus truncation or downloaded
pretrained checkpoint is involved. Hash the generated split, config, code and
initial tensors before execution.

Use one causal Transformer block, width 64, four 16-dimensional attention heads,
feed-forward width 256 and ReLU activation. Input tokens are a,b,equals; input
vocabulary 32, learned positional embeddings for three positions, and a separate
31-class readout at equals. Use pre-LayerNorm residual sublayers, a final
LayerNorm, LayerNorm epsilon 1e-5, no biases in attention/readout/MLP linears,
untied input/output weights, and no dropout. Embeddings/positions and linear
weights use independent normal initialization with standard deviation0.02;
LayerNorm scales start at 1 and offsets at 0. This explicit architecture is
paper-inspired, not a claim to reproduce unspecified author implementation
details. Preserve this distinction in any report.

Seeds0..9 vary initialization only. Within each seed, serialize one initial
state and reload identical tensors for every arm. All conditions use the same
GPU class, software versions, full-batch data order and parameter groups.
Record deterministic-operation settings; disable TF32 and mixed precision.
Do not change attention kernels between arms. Diagnostic RNGs must not advance
training RNGs or mutate optimizer state.

Full-batch AdamW: LR 1e-3, betas(0.9,0.98), epsilon 1e-8 outside sqrt, decoupled
weight decay1 on every trainable parameter, constant LR, no warmup or clipping.
FP32 parameters, gradients and optimizer moments in every arm. Use the same
unfused optimizer implementation. Train for 10,000 updates regardless of
accuracy; this is 300,000 updates across30 new trajectories, matched in data
exposure and optimizer boundaries. A batch has480 supervised targets and 1,440
input token positions; each trajectory has4.8M supervised targets. Adam
hyperparameters and update horizon follow the paper's small modular task;
the architecture and FP64-loss-only control are explicit choices here.

| Arm | Only varying factor | Purpose |
| --- | --- | --- |
| A | Native PyTorch CE on FP32 logits | Numerical baseline |
| B | Differentiable fixed hard-target CE in FP32 | Proposed intervention |
| C | Native CE on FP64-cast model logits; gradient cast back through the cast | Independent loss-arithmetic control |

C keeps the entire model forward, model backward, parameters and Adam state in
FP32. It is **not** a full FP64-training condition. The second FP64 precision
cliff still applies to C; measure whether it is approached. Fix only hard-target
loss arithmetic, without zero-sum projection, StableMax, custom detached-gradient
surrogates, altered temperature or gradient rescaling.

## Qualification before interpreting a null result

First perform CPU math/double-backward checks, then a bounded full-shape GPU
mechanics/timing calibration using dedicated calibration inputs/seed 1000.
Do not include that seed in the scientific cohort or tune using held-out
accuracy. Time all three loss arms, FP64 whole-model HVPs, an ordinary and an
Adam-scaled eigenpair solve, checkpointing and transfer. The calibration is
mechanics, not evidence. A failed calibration does not authorize a changed task
or epsilon: retain it and revise the prospective plan explicitly.

At every training state, use the actual installed CE backend on detached logits
to measure target-logit gradients. Log the fraction with native target gradient
exactly zero while the FP64-loss reference target gradient is nonzero and at
least one native non-target gradient is positive. Also log the relative
zero-sum defect sum(g_z)/sum(abs(g_z)), null when its denominator is zero.
Use the paper's log-sum-exp margin, not a top-two logit gap. Log margin quantiles
and fractions beyond nominal fp32/fp64 cliffs as descriptive context; measured
backend collapse, rather than a universal16.6 threshold, determines exposure.

For a test of the collapse-related mechanism, arm A must have >=10% collapsed
training examples at >=100 of its last 1,000 gradient states in at least 8/10
seeds. This exposure threshold is a proposed operational choice, not a paper
claim. B should restore the missing target gradients and agree with C on
same-logit gradient checks. C must not show meaningful FP64 collapse: report
any occurrence and withhold its reference interpretation for affected states.
Without exposure or a working reference, a null result is an unqualified test,
not evidence that the fix never changes curvature. Report every seed and failure;
do not replace seeds that fail qualification.

## One common curvature objective

At diagnostic states clone the actual FP32 parameter tensors exactly into a
separate FP64 model. Evaluate the entire480-example training objective F with
FP64 activations and differentiable fixed CE. No subsampling, dropout, optimizer
step, or diagnostic-induced change to training parameters. Independently compare
native FP64 CE values/gradients/HVPs where far enough below its measured cliff;
check saturated cases against analytic CE and stable finite differences.
ReLU gives a piecewise-smooth objective; the local Hessian is evaluated away
from activation boundaries. Finite steps can cross those boundaries, so retain
the measured Taylor remainder and do not assume the draft's smooth local
approximation holds over an entire Adam update.

Do not compute B's Hessian using B's training precision and A's using A's.
Do not only cast the final loss to double when claiming a fully FP64 diagnostic.
Do not reuse the paper's soft-target stop-gradient surrogate for double backward.

For H=nabla^2 F on all unique trainable parameters record:

- Primary: lambda_max(H), the largest **algebraic** eigenvalue. Matrix-free
  Lanczos/ARPACK with `which=LA`, deterministic start, tol 1e-3, ncv 20,
  maxiter 200; one unchanged-objective retry ncv 60/maxiter 400. Check a fresh
  eigenpair residual <=0.005. Unconverged solves are missing failures,
  never silently accepted or replaced with a random Rayleigh quotient.
- Gradient-direction curvature: kappa_g=g^T H g/||g||^2, using g=nabla F
  from the common diagnostic. This is the Hessian quantity closest to the
  draft's directional curvature; its adjacent-gradient proxy is not substituted.
- Actual-update curvature: kappa_delta=delta^T H delta/||delta||^2,
  where delta=theta[t+1]-theta[t] is the actual rounded training update,
  including decay. Evaluate H at theta[t], and save theta[t+1] as required.
- Adam-scaled maximum curvature lambda_max(D^(1/2) H D^(1/2)), where
  D=diag(1/(sqrt(vhat)+epsilon)) from that state's stored moments. This differs
  from both H and the beta1-scaled P convention in Analysis019. Save the
  definition with every output. State0 is undefined because there is no accepted
  moment state. This is a local descriptive quantity, not an Adam stability bound.

Use the same eigenpair residual discipline for both operators. Log eigensolver
time/calls. All norms/dots/prefix reductions use FP64; zero-gradient/update
denominators are null with a reason, with no arbitrary epsilon added.

Evaluate ordinary/scaled spectra and directional curvature at states
0,500,...,10000 plus9250 and 9750. The primary late window uses five states:
9000,9250,9500,9750,10000. These are equally weighted checkpoint summaries,
not a claim to measure every curvature spike between checkpoints. Capture the
next update at each nonterminal diagnostic; terminal update-direction curvature
is undefined and must not borrow a previous update.

Each update logs unsmoothed native optimization loss and a common accurate
FP64-CE-on-detached-FP32-logits loss, accuracy over all 480 training pairs, margins,
collapse diagnostics, raw gradient/momentum/adaptive/actual-update norms and
adjacent cosines. Keep A=sum norms, B=norm(sum vectors) and B/A for each stream,
with clear vector definitions. These scalar summaries do not retain arbitrary
gradient histories. Preserve first/zero-denominator nulls.

At curvature states, log full FP64 F, parameter norms by layer, gradient norm,
first-order predicted loss change, quadratic update term, and actual common-F
change for the next accepted update. Save the remainder. On Adam, never call
1-rho*norm(g_next)/norm(g_previous) an isolated Hessian estimate.

Evaluate all 481 held-out pairs at curvature states, with mean FP64 fixed CE and
accuracy. These do not select seeds, budgets, checkpoints, or learning rates.
Training loss curvature and held-out loss curvature are different claims; the
primary Hessian is explicitly a training Hessian.

## Primary decision and confidence/scale controls

For each seed and arm, L is the geometric mean of lambda_max(H) at the five
late-window states. The primary paired effect is log(L_B/L_A), averaged over
the ten independent seed blocks and exponentiated to report a ratio. Report
all ten paired ratios, a two-sided95% paired t interval in log space, and an
exact paired sign-flip sensitivity check over 2^10 assignments. Repeated
checkpoints are not independent replicates. The interval is a small-sample
model-based interval, not a guarantee of adequate power. Nonpositive or
numerically unresolved eigenvalues cannot be forced positive for logging;
report them and withhold the planned log-ratio verdict if necessary.

Predeclare20% as the smallest curvature reduction worth this first experiment:

| Qualified result | Interpretation |
| --- | --- |
| Entire95% ratio interval below 0.8 | Supports material raw-Hessian flattening in this task |
| Interval entirely above 0.8 | Rules out a20% average reduction at this confidence level; inspect direction of effect |
| Entire interval inside[0.8,1.25] | Stronger evidence of approximate equivalence within that band |
| Interval overlaps0.8 and extends outside the equivalence band | Inconclusive at the planned precision |

Reference-arm comparisons and all secondary measures are reported separately;
they do not replace a failed primary endpoint. C should have the same direction
of effect against A, and B/C should show broadly comparable outcomes. Disagreement
requires explanation, not an unsupported claim of a clean numerical mechanism.

To check whether raw flattening merely accompanies lower loss/higher margins,
retain first crossings of common accurate training-loss levels
1e-2,1e-3,1e-4,1e-5,1e-6. Save the actual parameter state at crossing and compute
the common FP64 Hessian offline. Compare A/B and B/C only when the evaluated
FP64 losses are within 10% multiplicatively; otherwise mark that level unmatched.
Do not choose the closest favorable crossing, extrapolate, or extend training
to manufacture overlap. Report lambda/F as a descriptive confidence control,
not a scale-invariant flatness metric. Layer norms/margin distributions disclose
remaining parameterization and confidence differences.

If fixed CE lowers raw curvature only at lower attained loss, the supported
claim is "lower curvature at matched updates, accompanying lower loss".
If the reduction persists at matched loss, that is stronger evidence for a
changed trajectory geometry. Neither comparison removes every rescaling
symmetry. If ordinary curvature is unchanged but Adam-scaled curvature or
finite-step penalties improve, report optimizer-conditioned smoothing.
If spikes disappear without either change, report a numerical-dynamics fix
without demonstrated flattening. A smoother curve alone cannot support H022.

## Runtime, cost, retention and stopping

Live Runpod `list_gpu_types(include=AVAILABILITY, product=POD, cloud=SECURE,
count=1)` read at 2026-10-23 approximately 15:30UTC: RTX 3090,24 GB,
USD 0.50/hour, LOW stock in EU-CZ-1. An RTX 4000Ada,20 GB, is also listed with
LOW stock in EU-RO-1 at USD 0.28/hour; do not silently substitute hardware after
timing qualification. The preferred3090 is an available economical class, not
a measured fastest/dollar ranking. Snapshot: [pricing.json](pricing.json).
Recheck price/stock before any eventual launch.

Use one worker; no multi-GPU, pretrained downloads or network volume needed.
The1-3 hour planning scenario is unmeasured and explicitly conditional: tiny
sequence length and dataset keep model work small, but FP64 eigenpairs and
Python dispatch can dominate. Historical Pythia and squared-error CIFAR
throughput are not this recipe's calibration. Measure a prospective ETC from
all 30 trajectories, calibration, diagnostics, saving and retrieval before
admitting production. If it cannot fit3.5 hours including retrieval, retain the
calibration and stop for redesign; do not silently cut seeds or diagnostics.

Proposed total envelope: USD 3, maximum4 hours from billable creation,
including setup/failures/retrieval. GPU arithmetic is 4*0.50=USD 2.00.
Allocate20 GB container disk plus10 GB pod volume, no persistent network volume.
Runpod's [storage documentation](https://docs.runpod.io/pods/storage/types),
read October 23, lists both running disks at USD 0.10/GB/month. At a720 hour
accounting approximation,30 GB for 4 hours adds USD 0.0167; actual provider
proration/invoice can differ. Reserve the remaining envelope for such charges;
do not treat USD 2.0167 as a posted bill. Recheck total resolved price and decline
creation if the maximum no longer fits USD 3. Owned local hardware is an
alternative with zero GPU rental; timing and interference still need checking.

Pin environment/code/config identities before scientific execution. Use a
detached sequential queue with manifest, persistent logs, events and watchdog.
Monitor read-only every 10 minutes, reporting completed seeds/updates/targets,
common loss, collapse fraction, eigenpair residuals, throughput, remaining time,
memory and retained bytes. Warn on nonfinite values, failed residual checks,
stale progress, disk pressure, and deadline/cost exhaustion. Diagnostic failure
is visible and never hidden by changing the scientific inputs.

Retain exact initial state per seed; scalar histories for every update;
all diagnostic and loss-crossing parameter states; next-update states needed
for Taylor checks; final recovery state; and rolling recovery every 250 updates
with optimizer moments, RNGs, data position and diagnostic prefix state. Save
code/config/environment/split identities. Cap evidence at 5 GB; there are no
dense per-update parameter or gradient histories. If measurements indicate the
cap is insufficient, stop before deleting required evidence or changing cadence.

At3.5 hours admit no more scientific work and finalize/retrieve by 4 hours. Use
independent remote training-stop and local retrieval/teardown guards. Verify
local sizes and SHA256 inventory before deleting the task-created pod. Required
evidence must not be abandoned to meet a nominal successful-run claim: preserve
it and report an incomplete run/recovery need if verification fails. Delete
the pod and confirm deletion after verified retrieval; leaving a stopped pod
retains billed storage. Report posted/provisional billing and any retained
resources. Manage no unrelated Spark or Runpod resources.

## Deliverables and limits

The eventual output is one compact PDF with common accurate loss, collapse
exposure, raw/directional/Adam-scaled curvature, and paired seed effects;
matched-loss results and full per-seed data accompany it. Each figure receives
an observation with sources, reductions, coverage and uncertainty. Report
qualification failures and negative/inconclusive outcomes. Insert no empirical
manuscript claim until the experiment exists and its evidence is verified.

The proposed experiment can identify a substantial effect in this deliberately
collapse-exposed small task. It does not establish that the original
CIFAR/ResNet/WikiText EOS patterns are numerical artifacts, that all directions
or held-out losses become flatter, or that fixed CE will improve Pythia.
Only a later direct matched CE experiment could support those transfers.
