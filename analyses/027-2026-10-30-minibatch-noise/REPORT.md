# Mini-batch noise at matched training states

Completed 2026-10-30. H023; calibration Run185 and production Run186.

The experiment supports a useful, bounded contribution: consecutive sampled
gradients describe both sampling variability and motion through the objective.
Their cosine and gradient-change proxy cannot be interpreted as isolated
full-gradient geometry when noise is appreciable. Frozen controls identify the
sampling contribution; complete gradients along SGD trajectories expose the
additional effect of parameter motion.

## Comparison and coverage

We use the first 5,000 CIFAR-10 training images, the existing tanh MLP
(3,072–200–200–10), and half-squared error summed over outputs and averaged over
examples. Normalization uses the full training split. Three initializations each
supply a new 2,000-update full-GD reference at constant learning rate 0.02. Saved
states after 0, 1,000 and 2,000 updates are shared exactly within each batch
comparison. Batch sizes are 32, 128, 512, 2,048 and 5,000.

Batches are uniform subsets without replacement, independently redrawn at every
evaluation. Common permutation prefixes pair batch sizes within a repetition.
There is no epoch-wise reshuffle, momentum, clipping, weight decay, schedule,
augmentation or dropout. Parameters and gradients are FP32, diagnostic reductions
are FP64, and TF32 is disabled. Full gradients, losses, covariance traces and
directional HVPs cover the entire 5,000-example objective.

At each saved state, 32 independent sequences of 128 frozen draws yield 184,320
observations, including repeated deterministic full-gradient controls. Eight
200-update continuations per stochastic batch and one full-GD control yield
297 forks and 59,400 updates. The 6,000 reference updates are additional. Every
moving state has a complete-gradient and loss reference; four state-matched HVP
pairs per fork give 1,188 independent derivative evaluations. “Independent” here
means computed by HVP rather than inferred from the proxy, not statistically
independent observations along a trajectory.

Historical seed-0 reconstruction **does not qualify** under the predeclared
scalar tolerances: the first cosine discrepancy exceeding 0.02 occurs at update
1,135. The paper therefore identifies new reference trajectories. This does not
invalidate their matched starting states. All six full-GD continuations from
states 0 and 1,000 reproduce their corresponding new reference segments exactly
in recorded loss, norm and cosine. See [verification.json](verification.json).

## Results and interpretation

The [frozen calibration](figures/frozen_calibration.pdf) closely matches the exact
finite-population variance and expected unnormalized noise-projection targets:
maximum relative errors across the 36 stochastic cells are 0.50% and 0.74%.
The largest absolute error of the concentrated-norm cosine approximation is
0.00745. That error is small but detectable: the prediction lies outside the
conditional 95% bootstrap interval in 21/36 cells. Norm coefficients of variation
range from 0.018 to 0.135. Thus the data delimit the approximation rather than
establishing an exact expectation-of-a-ratio law.

Signal strength matters as much as nominal batch size. At update 1,000, even
batch 2,048 gives noise-to-signal squared-norm ratio `V_b/S` between 13.37 and
13.57 across the three initializations. A large batch fraction does not ensure
negligible relative noise at a trained state.

The [moving geometry](figures/sgd_geometry.pdf) changes the interpretation further.
Starting at initialization, batch 32 gives mean sampled cosine −0.141, while
complete gradients along those **same SGD states** have mean cosine +0.353.
The separate full-GD control has mean cosine +0.994. Noise and the changed path
both matter; the two cosine streams are not an additive decomposition.

From state 2,000, batch 32 moves sampled cosine toward zero, from the full-GD
value −0.991 to −0.384. Nevertheless, endpoint normalized displacement remains
about 0.03 in both conditions. Mean full-objective loss increases by 0.0241 under
batch 32 and decreases by 0.0087 under full GD. The stochastic increase is positive
in every initialization-specific mean. Conversely, batch 2,048 ends below the
full-GD loss in all three blocks at this fork. There is no universal monotonic
batch-performance result. [Full loss histories](figures/sgd_full_loss.pdf) and
[paired endpoint contrasts](tables/analysis027_contrasts.tex) retain these effects.

The strongest validation concerns the curvature interpretation. At batch 32,
the median absolute discrepancy between the raw proxy `q = eta kappa_hat` and
the state-matched directional HVP is 0.7467. Measuring the explicit noise term
and comparing `q + epsilon` reduces it to 0.0030 across 288 probe pairs. The
corresponding corrected 90th percentile is 0.0143. All stochastic batches show
substantial improvement. [The proxy figure](figures/sgd_proxy_validation.pdf)
separates smoothed context from raw paired markers; [the HVP table](tables/analysis027_hvp_errors.tex)
uses individual unsmoothed pairs. The correction identity is algebraic; the
separate HVP calculation tests the remaining finite-step approximation. This is
an offline reference requiring complete gradients, not a free correction
available from two sampled gradients.

## Uncertainty and limits

Frozen intervals resample entire sequences, preserving adjacent-pair dependence.
Moving conditional intervals resample whole continuations at fixed initialization
and checkpoint. All use 4,000 bootstrap replicates. Plotted moving bands are the
minimum/maximum of three initialization means, **not confidence intervals**.
The nine saved states share three reference trajectories and are not nine
independent initialization replicates. Temporal summaries use all 199 defined
pairs; 20-pair trailing means are only a display choice. Raw companions are retained.

This is one MLP objective, one learning rate and 200-update forks. Checkpoints
were selected using earlier exploratory evidence; common indices do not imply
common dynamical regimes. Equal updates entail unequal data exposure. Neither
cosine sign nor normalized displacement establishes stability, convergence,
generalization or an LLM batch requirement. No validation/test claim is made.
For later large-model work, measure relative noise at the evaluated states and
distinguish raw gradients from optimizer directions.

## Evidence, manuscript and execution

[Observation001](observations/001-controlled-minibatch-noise.md) specifies every
figure's sources, reduction, caption, uncertainty and caveats. The
[figure review](FIGURE_REVIEW.md) and [manuscript integration](observations/002-manuscript-integration.md)
record the original visual and numerical checks. The subsequent
[focused editorial revision](observations/003-focused-manuscript-revision.md)
replaces that presentation with two batch-size figures and essential methods
in Section V. The added Appendix E, dense tables and HVP discussion are removed
from the paper; all original diagnostic evidence remains in this report.
Section III and the supplied original assets are preserved.

One task-owned Runpod RTX 4090 at the live quoted rate of USD0.89/hour completed
production in 1,371.56 seconds (22.86 minutes). The full provision-to-cleanup
window was 57.90 minutes. Estimated compute plus storage is **USD0.87 provisional**,
below the conservative USD5 guard; posted billing had not arrived at closeout.
All 7,178 inventory-listed files (15.192 GB) were retrieved and size/SHA-256
verified before deletion of the task pod and 30 GB volume. Cloud storage retained:
zero. Spark and unrelated resources were untouched. Raw evidence remains local
under the owning runs, outside Git. The verified compressed transfer archive is
also retained in ignored cache (10.267 GB). See [retrieval](retrieval.json) and
[closeout](closeout.json).
