> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-gradient-amplification.md](../../../../../../analyses/062-2026-11-07-mlp-gradient-amplification/observations/001-gradient-amplification.md).

# Observation: growing output oscillations and reduced gradient cancellation

Run237 / Analysis062 / H042, 7 November 2026. Exploratory mechanistic follow-up
to the user's RMSNorm hypothesis. This observation concerns the unchanged seed-0
CIFAR10 MLP Adam control; it does not establish a mechanism for other networks.

## Conclusion

There is no RMSNorm, LayerNorm or BatchNorm in this MLP. The RMS values in the
previous analysis were diagnostics, not operations in its forward pass. The new
evidence favors an oscillation whose amplitude grows along already sensitive
network directions. Predictions reverse direction on successive updates,
individual-example errors become less cancelling in parameter-gradient space,
the batch gradient grows disproportionately, and actual Adam increments grow.
The resulting output displacement can overshoot even though every measured
increment is initially downhill. This is a measured local mechanism, not a
unique identification of what first excites or sustains the oscillation.

No sudden collapse in Adam's denominator is observed. This does **not** rule out
Adam's existing preconditioning or momentum as contributors to the dynamics.
The frozen-Jacobian probe shows that the late update points along directions
that were already more sensitive in the earlier network. It does not estimate
the top Jacobian singular value or the full stability spectrum.

## Comparison and coverage

The architecture is Flatten -> Linear(3072,200) -> Tanh -> Linear(200,200) ->
Tanh -> Linear(200,10), with biases: 656,810 parameters. Full objective
L = (1/2) mean_i ||f_i - onehot(y_i)||^2 on the same pinned 5,000 training
examples; physical batch 1,000. Adam LR .001, betas (.9,.999), epsilon 1e-8,
no weight decay or clipping; FP32 model and diagnostic vectors, FP64 reductions,
TF32 disabled. No change to initialization, data ordering, optimizer or policy.

All 536 loss/gradient-norm pairs from state 0 through 535 exactly match Run223.
All 15 retained Run236 model/optimizer/RNG/current-and-previous-gradient anchors
match. Observers preserve RNG and parameter gradients. The full diagnostic
window is 448..535 (88 states, 87 outgoing increments); selected events are
465/474/485, 492/493/508, 502/503/505, and 512/522/529 (before/peak/recovery).
Peak503 is the literal last pre512 local peak. The two larger peaks provide
clearer signal but are not independent replications. No examples are truncated.

## What spikes together?

| Quantity | State465 -> peak474 | State512 -> peak522 |
|---|---:|---:|
| Full loss | .028188 -> .053673 | .023057 -> .055063 |
| Batch gradient norm | .4881 -> 2.0176 (4.13x) | .4096 -> 2.1839 (5.33x) |
| RMS per-example gradient norm | 4.098 -> 6.666 (1.63x) | 3.722 -> 6.993 (1.88x) |
| Gradient coherence | 1.419% -> 9.162% (6.46x) | 1.211% -> 9.752% (8.05x) |
| RMS parameter Jacobian norm | +0.50% | +0.36% |
| RMS input Jacobian norm | +0.87% | +0.68% |
| Examples whose loss increases | 89.70% | 92.90% |
| Cosine of incoming and previous output changes | -.9851 | -.9900 |

The unit gradient is g_i = grad_theta[||r_i||^2/2], without the batch 1/N factor.
Coherence C = ||mean_i g_i||^2 / mean_i||g_i||^2. This measures cancellation,
not the unweighted average pairwise cosine or the lag-one gradient alignment.
The exact identity ||g|| = sqrt(mean_i||g_i||^2) sqrt(C) decomposes the norm
growth. For the first event, 1.6266 * sqrt(6.4570) = 4.1332. Per-example gradient
energies are measured through the layer outer-product identity, not inferred
from a batch-mean gradient. Bias contributions are included.

The effect appears in all three weight matrices: their batch gradient norms
grow about 4.04x/4.23x/4.04x at peak474, versus individual-gradient RMS growth
1.74x/1.44x/1.38x. It is not confined to the final layer. Increased output error
and reduced cancellation occur together; these observations alone do not give
an arrow of causality between the two.

The 1% of examples with the largest positive loss increases account for only
7.41% and 6.40% of total positive increase at the two large peaks. The strongest
single first-hidden-unit backpropagation energy share is 5.18% and 5.58%; the
top 20 of 200 units account for 43.47% and 48.72%. There is concentration, but
neither a single runaway example nor a single runaway unit explains the spike.

## Tanh and backpropagation

At state465/peak474, the observed vector-norm gains through the backward chain
are: head .777/.814; tanh2 .883/.883; W2 1.236/1.257; tanh1 .302/.346;
W1 1.146/1.141. Thus both tanh operations attenuate the actual backpropagated
vectors. First-layer attenuation weakens somewhat as the residual changes
direction, without a singular derivative. The derivative 1-tanh(z)^2 lies in
[0,1]; it equals one near zero. Capped activations do not fix their signs,
directions or coordinated changes, and do not cap the loss of the linear head.

The preceding [Analysis061 observation](../../../../../../analyses/061-2026-11-07-mlp-adam-spike-internals/observations/001-spike-internals.md)
found almost unchanged hidden RMS/parameter norms, small bias/head-parameter
contributions in static recombinations, and predominantly output-sensitive
hidden-feature displacement. The new backward and directional probes support
that interpretation; they are additional diagnostics of the same trajectory.

## Why can a downhill update increase loss abruptly?

For this squared-error objective there is an exact output-space identity:

    L(s+1)-L(s) = mean_i r_i^T Delta_f_i + 0.5 mean_i ||Delta_f_i||^2.

For update473 -> 474 the two measured terms are -.0926296 and +.0985978,
leaving +.0059682. For update521 -> 522 they are -.1153251 and +.1214801,
leaving +.0061549. Output displacement is large enough that its quadratic cost
exceeds the residual correction. Successive output changes almost reverse,
and projections on each selected before-to-peak displacement show alternating,
growing excursions. These projection directions are fitted to their events
post hoc; the near-negative-one consecutive-vector cosines do not depend on
that projection choice.

The local network prediction, g^T Delta_theta + 0.5 mean||J Delta_theta||^2,
tracks the 87 actual loss changes with RMSE .0002647, versus loss-change RMS
.0045792; signs agree for 85/87. All 87 actual increments have negative
g^T Delta_theta, yet 42 increase loss. This concerns the actual parameter
increment, not an unpreconditioned momentum or gradient-alignment statistic.

Full second-order parameter Taylor expansion also includes
0.5 mean[r_i^T f_i'']. Its RMSE is worse here (.0007212). It is retained and
reported, not discarded: higher-order finite-step terms matter and can cancel
the local residual-curvature contribution. The exact output identity above
does not use a Taylor approximation.

## Fixed network versus changed update direction

The post hoc crossing evaluates the full-example directional gain
sqrt(mean||J Delta_theta||^2)/||Delta_theta||. It reconstructs each actual Adam
increment from the exact stored model, optimizer and current gradient.

| Jacobian state | Direction465 | Direction473 |
|---|---:|---:|
| 465 | 4.1750 | 5.7635 |
| 473 | 4.1947 | 5.7510 |

The incoming increment grows from norm .02878 at465 to .07811 at473 (2.71x).
Its directional gain grows about 1.38x. Almost the same directional gain is
obtained with the earlier Jacobian: changing the network while holding the
late direction fixed changes gain by -.22%. At the second large event,
direction512/521 gains on the state512 Jacobian are 4.0614/5.6920, compared with
4.0463/5.6925 on the state521 Jacobian. Late-step gain changes only +.009% when
the Jacobian changes. Its step norm grows .02642 -> .08817 (3.34x).

The evidence therefore favors growing increments directed into existing
sensitive directions. It does not require an abruptly exploding Jacobian.
Trace stability alone would not establish this; the crossed-direction probe
tests the actual directions explicitly.

## Small denominators: RMSNorm and Adam

Standard RMSNorm uses x/sqrt(mean(x^2)+epsilon), optionally multiplied by gamma.
A single near-zero coordinate does not imply a near-zero normalization scale.
For unit gamma, its input Jacobian is I/r - xx^T/(n r^3), where
r^2=mean(x^2)+epsilon. Near an all-zero vector, the gain approaches
1/sqrt(epsilon): finite but potentially large. With a gain vector, an operator
norm bound is max|gamma|/sqrt(epsilon). Thus the quoted concern is valid in a
network that has this operation, but a large local gain alone does not establish
an exploding end-to-end update. Formula reference: [PyTorch RMSNorm documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html).
This reference defines the standard operation; installed Spark PyTorch is2.11.

This MLP has no such operation. Its cached input normalization is fixed; the
smallest scale is .22517, not near zero. Adam has a different division,
D_s=sqrt(vhat_s)+epsilon, which was measured coordinate by coordinate. Across
the whole window its smallest value is2.1605e-5 (>2,160 epsilon); no coordinate
is within10 epsilon. The largest one-step decrease is only .1383%. With beta2
.999 and nonnegative squared-gradient additions, the bias-corrected denominator
has a one-step lower ratio bound sqrt[beta2(1-beta2^s)/(1-beta2^(s+1))]; adding
epsilon preserves this lower bound. The measured ratios satisfy it to FP32
roundoff. No epsilon-floor or denominator-collapse event occurs.

Using the previous denominator with the new momentum changes the resulting
increment norm by only -.096% to +1.280% across the window (static probes, not
retrained trajectories). This excludes a sudden denominator collapse as the
immediate explanation; it does not exclude the effect of the existing adaptive
scales or their slow response over multiple steps.

## Figures, sources and reproducibility

- `figures/01-gradient-amplification.pdf`: complete448..535 diagnostic window.
  Loss, relative batch/per-example gradient strength, coherence, RMS Jacobian
  sensitivities, unit-loss backpropagation, actual increment norm, denominator
  changes and consecutive output-increment cosine. Dotted lines mark the four
  selected peaks. Baseline-normalized curves use465. Update quantities at s
  refer to s -> s+1; the output cosine compares s-1 -> s with s -> s+1.
  The denominator-change RMS is weighted by squared actual W1 increments;
  the minimum change is over all parameters. Different axes have different
  scales. Jacobian RMS means sqrt(mean squared Frobenius norm), not spectral norm.
- `figures/02-oscillation-and-direction-probes.pdf`: columns for larger peaks474
  and522. Rows show signed output projections, actual/local predicted loss
  change, the two **exact** output-space loss terms, and the crossed Jacobian/
  increment directional gains. Projection is centered at the preceding state
  and divided by sqrt(5000). The direction is the selected before-to-peak full
  50,000-coordinate prediction displacement. Last-row bars are static probes.

Generator: `01_report.py`; post hoc probe: `02_cross_jacobian_probe.py`.
Raw sources: Run237 `artifacts/spark/payload/worker/{backward,adam,directional,
boundaries}.jsonl`, `output_vectors.pt`, `backward{state}.pt`, and the retained
Run236 snapshots. Hashes, reductions and plotted values are in `results.json`,
`plot_data.json`, `input_manifest.json`, transfer inventory and probe metadata.
`03_verify.py` independently checks all536 reference pairs, final535 full state,
57 input/28 artifact hashes, all87 directional reductions,15 per-example-energy
anchors, four actual consecutive increments and all10 CPU/GPU crossed probes.
The largest CPU/GPU crossed-probe relative difference is1.01e-7. Prelaunch
double-precision tests check formulas against autograd Jacobians/Hessians.

No statistical confidence interval is appropriate for this deterministic,
single-seed selected window. Floating-point agreement is not uncertainty over
seeds, datasets or alternative trajectories. Full arrays remain retained;
the observed feedback pattern does not identify a unique initiating cause or
establish that changing beta2/momentum alone will fix it. No manuscript claim
or production optimizer change is made here.
