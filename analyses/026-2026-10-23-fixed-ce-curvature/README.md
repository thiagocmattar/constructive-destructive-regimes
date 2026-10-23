# Analysis026: does numerically fixed cross-entropy lead to flatter curvature?

Status: prospective experiment proposal, October 23,2026. H022 is the user's
intuition, operationalized below. No training run or cloud resource is allocated,
and the proposed spending limit is not an executed charge. No manuscript edit.

## Recommendation

Use a tiny modular-addition Transformer, ten paired initialization seeds, and
three cross-entropy arithmetic conditions: native FP32, fixed FP32, and native
CE evaluated in FP64 on the same FP32 model logits. Train each for 10,000
full-batch AdamW updates. Evaluate every condition's curvature with the same
fully FP64 model copy and differentiable fixed hard-target CE. This separates a
change in the learned parameters from an error in the curvature measurement.

The primary question is whether the fixed-loss trajectory has a materially lower
largest algebraic training-Hessian eigenvalue over the last 1,000 updates. Also
measure gradient-direction curvature, actual-update curvature, and Adam-scaled
curvature. Report matched-loss comparisons and parameter norms to qualify
confidence and coordinate-scale effects. Smoother loss alone is not flattening.

The proposed cloud envelope is one RTX 3090 for at most 4 hours, USD 3 total,
including setup, calibration, diagnostics and retrieval. Runtime is unmeasured
for this recipe; the planning scenario is 1-3 hours, conditional on calibration.
Owned hardware is also suitable and incurs zero GPU rental. See
[PROTOCOL.md](PROTOCOL.md) for the comparison, decision rules and cost guards,
and [pricing.json](pricing.json) for the live catalog evidence.

## Why this experiment

The linked [paper, v1](https://arxiv.org/pdf/2610.31642v1), fixes numerical
evaluation of CE; its hard-target formula has the same exact-real objective.
Consequently it cannot change the exact Hessian at fixed parameters. Different
optimization trajectories can nevertheless reach different curvature. The
paper's modular-addition setup deliberately reaches the high-confidence regime,
making it more informative per dollar than a large language-model replay whose
loss implementation may never encounter substantial collapse.

The working [draft Section3](../../manuscript/draft/sections/03-methods.tex)
distinguishes directional curvature from consecutive-gradient alignment. The
existing evidence supplies useful machinery and cautions:

- [Analysis019](../019-2026-10-13-cifar10-eos-lr/README.md): ordinary and
  Adam-scaled Hessians, but its CIFAR MLP uses squared error. Its trajectories
  are not CE controls.
- [Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md): larger-model
  Hessians cover subsets and the Transformer is far more expensive. The tiny
  experiment can cover its entire training objective.
- [Analysis025](../025-2026-10-17-pythia-descent-mechanism/REPORT.md): actual
  adaptive-step effects can differ from first-order predictions. This motivates
  measuring the update direction and Adam scaling, rather than treating raw
  gradient alignment as an Adam curvature measurement.
- [Analysis016](../016-2026-10-11-rosenbrock-hessian/README.md): better attained
  loss can coexist with higher maximum curvature. The anticipated sign needs a
  test, not an assumption.

## Verification and scope

`python analyses/026-2026-10-23-fixed-ce-curvature/01_check_loss_mechanics.py`
checks the differentiable fixed loss, gradient and full logit Hessian against
analytic binary CE, including a saturated FP32 example. The compact result is
[loss_mechanics.json](loss_mechanics.json). These are CPU mathematical checks,
not training evidence, a throughput calibration, or support for H022.

The experiment would be conclusive about a large curvature effect in this
specified task if qualification and uncertainty criteria pass. It cannot
guarantee a decisive outcome, establish flatness invariant to reparameterization,
or establish transfer to CIFAR/Pythia. An inconclusive result remains explicit;
there is no automatic seed, horizon, epsilon or task search.
