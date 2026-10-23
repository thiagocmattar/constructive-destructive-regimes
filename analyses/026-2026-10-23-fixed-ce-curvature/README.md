# Analysis 026: does numerically fixed cross-entropy lead to flatter curvature?

Status: complete and verified, October 23,2026. All 30 scientific conditions and
300,000 updates finished on the approved owned-GPU alternative. Run 154 owns
calibration and Runs 155-184 own the paired conditions. No cloud resource was
created and GPU rental is zero. No manuscript edit.

**[Conclusions and controls](REPORT.md):** fixed CE reduces the late ordinary
maximum-Hessian endpoint by 98.5% (paired ratio 0.0149,95% CI0.0031-0.0720).
Matched-loss curvature is similar; Adam-scaled curvature rises 2.4 x. Hard-collapse
exposure is below the prespecified criterion. The simplified
[FP32 publication figure](figures/fixed_ce_curvature_fp32.pdf) shows only native
and fixed FP32 training: loss, eta*lambda_max, measured eta*kappa, and the
manuscript's consecutive-gradient eta*kappa_hat proxy. The
[figure observation](observations/002-focused-fp32-figure.md) defines the two
kappa quantities and their AdamW interpretation. The original
[eight-panel control figure](figures/fixed_ce_curvature.pdf) and
[completed observation](observations/001-completed-experiment.md) retain all arms,
coverage, uncertainty and the narrower interpretation.

## Original prospective design

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

The linked [paper, v 1](https://arxiv.org/pdf/2610.31642v1), fixes numerical
evaluation of CE; its hard-target formula has the same exact-real objective.
Consequently it cannot change the exact Hessian at fixed parameters. Different
optimization trajectories can nevertheless reach different curvature. The
paper's modular-addition setup deliberately reaches the high-confidence regime,
making it more informative per dollar than a large language-model replay whose
loss implementation may never encounter substantial collapse.

The working [draft Section 3](../../manuscript/draft/sections/03-methods.tex)
distinguishes directional curvature from consecutive-gradient alignment. The
existing evidence supplies useful machinery and cautions:

- [Analysis 019](../019-2026-10-13-cifar10-eos-lr/README.md): ordinary and
  Adam-scaled Hessians, but its CIFAR MLP uses squared error. Its trajectories
  are not CE controls.
- [Analysis 024](../024-2026-10-17-eos-gd-lr-calibration/README.md): larger-model
  Hessians cover subsets and the Transformer is far more expensive. The tiny
  experiment can cover its entire training objective.
- [Analysis 025](../025-2026-10-17-pythia-descent-mechanism/REPORT.md): actual
  adaptive-step effects can differ from first-order predictions. This motivates
  measuring the update direction and Adam scaling, rather than treating raw
  gradient alignment as an Adam curvature measurement.
- [Analysis 016](../016-2026-10-11-rosenbrock-hessian/README.md): better attained
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

## Execution resolution

The owned RTX5070 Ti Laptop GPU passed full-shape calibration, using less than
0.12 GB allocated memory. The queue's measured estimate is about 48 minutes of
production work, about 73 minutes including 30% allowance and ten-minute closeout.
Cloud stock changed between read-only queries; the already-approved local
alternative avoids rental without changing scientific inputs. The detached
queue has 3.5 hour training admission and four-hour independent process guards.

The frozen raw metric schema uses `gradient_state` for the pre-update state,
so accepted update number is exactly `gradient_state + 1`. The `update` key
contains actual-displacement geometry; it supersedes the worker's initially
constructed integer field of the same name. No gradient, actual-update metric
or time index is lost, and reduction verifies the full contiguous state sequence.
Raw evidence and executed source remain unchanged.
The training `update` stream forms successive-parameter differences in FP32
before FP64 geometry reduction. Its accumulated displacement is checked against
the exact FP64 difference of the initial/final stored tensors, with roundoff
reported. Curvature along the actual update uses the FP64 difference of the
two stored parameter states directly.

After the first seed showed curvature separation without exact target-gradient
collapse, an exploratory checkpoint-gradient audit was specified in
`10_checkpoint_audit.py`. It compares three gradient implementations at the same
retained states, without updates. It cannot replace the predeclared endpoints
or collapse qualification and will be reported separately as post-hoc analysis.

## Closeout

The scientific queue completed at 18:55:37 UTC (15:55:37 Sao Paulo), after 40 min 18 s.
All 840 ordinary and 660 Adam-scaled eigenpairs,300 k scalar rows and paired input
identities passed verification. All calibration and production evidence totals
1,061,488,148 bytes locally, including required recovery states. The task
processes exited; no cloud compute/storage was created or retained.
See [artifact inventory](artifact_inventory.json), [output checks](output_verification.json)
and [closeout](closeout.json). Raw evidence and executed source are preserved.
