# Run234: MLP GD learning-rate extension

H040 / Analysis060. Authorized 7 November 2026: run plain full-batch GD at
LR0.04, then LR0.08, independently from original Run223 Adam state512 to800.
Retain Run233 LR0.02 unchanged. Each treatment starts with identical weights,
RNG, full5000-image CIFAR-10 MSE objective, physical batches1000, seed0 and
FP32 model/raw gradients; diagnostic reductions use FP64. Plain SGD optimizer
has zero momentum/weight decay and no adaptive scaling or clipping.

The retained Adam(.9,.999), LR.001 control is qualified by exact fork gradient
and two-update model/optimizer/gradient-history/RNG replay on the same Spark.
Each new LR also passes exact2 versus1+1 resumed updates before its288-update
scientific branch. Qualification branches reset; the .08 branch does not inherit
the .04 endpoint. The common GD trajectory code is promoted from frozen Run233
to src/eos_tasks/gd_intervention.py, now accepting an explicit LR. Run233 inputs
are preserved. Compare rates within GD; comparisons to Adam change optimizer
and LR jointly and cannot isolate optimizer effects.

Measure every boundary512..800: full-objective loss, raw norm, adjacent raw
gradient cosine rho and projected change eta*kappa_hat =
<g_(s-1),g_(s-1)-g_s>/||g_(s-1)||^2. The pair at512 crosses the fork history.
Zero denominators yield null. Every outgoing update records applied LR, actual
FP32 increment norm, raw-gradient dot/cosine and residual from -LR*g.
Retain first-step513, rolling32-update recovery and final800 checkpoints with
optimizer, RNG, completed state and gradient history. Full10000-image test
loss at800 is secondary. Main outcomes: mean513..800, final32 mean, final/max
loss and >10% fixed-anchor excursions with censoring. Single seed/window;
no convergence, generalization or universal stability claim.

Owned Spark only: expected under60 seconds including qualification (Run233
main branch measured2.78s); worker limit600s, detached timeout630s plus30s kill
grace. Logs/progress/manifest persist across SSH loss. Stop nonfinite loss or
gradients; do not retune failed branches. Expected retained evidence below0.4GB
locally and remotely; verify every file size/SHA before closeout. No cloud cost,
no unrelated process changes. Analysis060 owns updated figure and findings.

## Verified execution

Worker ended7 November at13:04:24UTC after4.159s. LR0.04 completed288 updates
in2.623s, reaching loss0.0588509 at800. LR0.08 applied35 updates in0.352s,
reaching finite loss6.0501e34 at546 before objective overflow at547. The worker
manifest truthfully remains `failed`; `verification_final.json` passes integrity
and update-rule checks for the completed and failed scientific outcomes.

All42 input identities and68 retrieved files plus archive pass size/SHA checks.
Exact Adam replay and both GD2 versus1+1 checks pass. `04_audit_failure.py`
replays three recorded updates from recovery544, exactly matching finite loss
and gradient norms through546, then reproduces infinite loss547 with finite
weights/gradients. It writes separate audit evidence; original artifacts remain
unchanged. Qualification includes10 initial updates plus3 audit replay updates.

At0.04, mean/tail training losses are394.45%/232.35% above Adam, with one
unrecovered excursion. No full-window or endpoint statistic is fabricated for
0.08. See [Analysis060 Observation002](../../analyses/060-2026-11-07-mlp-adam-to-gd/observations/002-gd-learning-rates.md).
The three-column figure preserves all old0.02 values. Maximum allocated CUDA
memory457,576,960 bytes. Worker exited;261,300,224 physical bytes retained on
Spark including archive; zero cloud cost. No unrelated processes changed.
