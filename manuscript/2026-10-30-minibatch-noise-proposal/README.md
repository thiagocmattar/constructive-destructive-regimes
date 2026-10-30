# Mini batch noise experiments and manuscript proposal

Date: 2026-10-30. Status: proposed scientific design; no training launched and
no changes applied to the working manuscript.

The strongest addition would establish **when consecutive-gradient geometry
reflects sampling variability, and when it reflects changes along the training
trajectory**. Use a frozen-state experiment to test Section III-B directly, then
matched SGD continuations to determine how those effects transfer to actual
displacement. A batch-size sweep of training curves alone cannot separate them.

## What the existing experiment establishes

The [Analysis020 figure](../../analyses/020-2026-10-14-eos-multitask/figures/eos_frozen_noise_comparison.pdf)
does use only initial parameter states, but includes **three initializations per
model**, not one initialization overall. Its [observation](../../analyses/020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md)
records three models, four batch conditions, 32 sequences of 128 gradient draws
per condition and initialization, and zero optimizer updates. The complete
training objective supplies every full-gradient reference. There is no Hessian
subset issue in this reference measurement.

| Case | Smallest sampled batch | Noise RMS / full-gradient norm | Mean cosine | Mean normalized accumulation |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 / MLP | 32 images | 1.232 | 0.398 | 0.639 |
| CIFAR-10 / ResNet-32 | One fixed 1,000-image BN group | 0.403 | 0.863 | 0.930 |
| WikiText-2 / Transformer | 8 sequences | 1.071 | 0.467 | 0.687 |

These are the recorded means across initializations, with within-initialization
sequence averaging. The figure's bands are ranges across three initialization
means, not confidence intervals. Its horizontal axis is measured relative noise,
not batch size. The ResNet result is informative: its smallest allowed batch is
still relatively signal dominated, so its cosine remains high.

Keep this experiment as a control. Its present nine-panel form has three limits
for the new main-text argument:

- It does not test later training states or actual SGD trajectories. Its
  `R_128` is normalized accumulation of sampled gradients at fixed parameters;
  actual parameter displacement is zero.
- It does not display the finite-population variance prediction, the expected
  negative noise projection, or the cosine approximation against measurements.
  Therefore it illustrates the mechanism more than it tests the three equations.
- Its bottom-row zero correction follows algebraically because the full gradient
  is unchanged. It is a useful implementation check, not independent evidence
  that the curvature estimate is accurate. No optimizer step or learning rate
  is used in that control.

For publication, retain its sources and values, put protocol details in the
caption rather than an embedded footer, label batch sizes, and move the complete
initialization-only view to the appendix once trained-state evidence is available.

## What the three equations mean

The current compiled draft, page 5, contains Eqs. (17)--(19). Let
`mu = gradient L(theta)`, `S = ||mu||^2`, and
`V_b = (N-b)/(b(N-1)) tr Sigma(theta)`, using the manuscript's population
covariance normalization `1/N`.

1. **Eq. (17) is an exact expected unnormalized noise projection**, under
   conditionally unbiased fresh sampling and finite second moments. Smaller
   batches increase its negative magnitude at matched parameters. This is not
   a theorem that the expected total interaction is negative, nor that the
   expectation of the normalized `epsilon_t` has the same sign.
2. **Eq. (18) is an exact conditional variance law** for uniform sampling without
   replacement inside each batch. Fresh batches across updates are different
   from partitioning one shuffled epoch: the latter introduces temporal
   dependence and does not satisfy the same conditional-unbiasedness argument.
   Across moving trajectories, batch size also changes the states and therefore
   `Sigma(theta)`; a universal ordering does not follow.
3. **Eq. (19) is a frozen/slow-motion, concentrated-norm approximation**:

   $$\mathbb E[\rho]\approx\frac{1}{1+\chi_b},\qquad
   \chi_b:=\frac{V_b}{\|\nabla\mathcal L(\theta)\|^2}.$$

   Smaller batches increase `chi_b`; cosine can then approach zero. The noise
   term itself does not attenuate toward zero. The predicted response should be
   computed separately for each checkpoint and initialization, before averaging:
   applying the nonlinear formula to a pooled mean noise level changes the target.

At a fixed state with independent draws, the expected adjacent inner product is
exactly `S`, because the extra `V_b` in the preceding gradient's squared norm
cancels the expected `-V_b` noise contribution. This makes the distinction between
the noise term and total interaction essential.

There is also an exact frozen-state identity, for nonzero independent identically
distributed gradients:

$$\mathbb E[\rho]=\left\|\mathbb E\left[\frac{g}{\|g\|}\right]\right\|^2\geq0.$$

It does not require concentrated norms, but does not supply the particular
`1/(1+chi_b)` response. For example, an equally likely scalar gradient population
`{3,-1,-1,-1}` has zero mean and expected cosine `1/4`. This shows why zero mean
or large noise alone is insufficient without the approximation's assumptions.
Persistent negative expected cosine in a matched frozen iid control would signal
a violation of its setup; negative cosine along moving SGD can be legitimate.

For normalized accumulation, independent frozen gradients satisfy
`E||sum_{t=1}^T g_t||^2 = T^2 S + T V_b`. Under additional concentration of the
norms and cumulative squared norm, a useful reference is

$$R_T\approx\sqrt{\frac{S+V_b/T}{S+V_b}}.$$

This is not an exact expectation of a ratio, and should be an appendix reference
unless its error is measured. In a noise-dominated random-walk limit its scale
is `T^(-1/2)`, not zero at a finite horizon. With a nonzero drift, the long-sequence
limit is `||mu|| / E||g||`. Neither result establishes a universal batch ordering
for actual SGD displacement.

## Recommended experiment

Use the existing first-5,000-image CIFAR-10, tanh 200-200 MLP as the mechanistic
test. Its complete-objective gradients are feasible to measure every step and
there is no batch-normalization context to confound a batch-size change.
Retain the existing three-model initialization control as broader context.

| Component | Proposed comparison | Scientific role |
| --- | --- | --- |
| Reference trajectories | Full GD, LR 0.02, seeds 0/1/2; states after 0, 1,000 and 2,000 updates | Supply matched initial, intermediate and later states |
| Frozen controls | At every state, batch sizes 32/128/512/2,048/5,000; 32 independent sequences of 128 draws | Test variance, noise projection and cosine approximation without motion |
| Moving continuations | From every state, same five batches, LR 0.02, 200 updates; eight sampling seeds per stochastic condition | Measure actual displacement and the interaction of noise with the evolving objective |
| Deterministic control | One full-GD continuation per state and initialization | Avoid treating identical deterministic repeats as replication |
| Reference diagnostics | Complete gradient and objective at every visited state; directional HVP at predeclared steps | Separate realized sampling noise, finite-step gradient change and local curvature |

The existing seed-0 LR 0.02 trajectory first reverses at gradient update 1,156,
which motivates the 1,000/2,000 checkpoints. This is an exploratory selection
based on existing evidence. Keep the same update indices for the other seeds;
do not relabel them as universally pre/post-instability states or move them to
obtain attractive curves. The existing Run062 output retains initial/final/rolling
recovery files, not the required intermediate states. Reconstruct and verify its
prefix, or identify a new reference trajectory if numerical reproduction fails.
The detailed [measurement and figure contract](01-experiment-and-paper-design.md)
specifies this distinction.

The most informative comparison is between `rho_t` from the sampled gradients
and the cosine between the **complete gradients at the same two SGD states**.
These full gradients describe the sampled trajectory; they are not a separate
GD trajectory. Their difference is descriptive, not an additive cosine
decomposition. Exact noise accounting instead uses unnormalized inner products.

Treat outcomes as tests, not requirements. Weak sampled alignment with coherent
full gradients supports sampling attenuation. Changes in both indicate a
trajectory contribution as well. A nonmonotonic response, persistent negative
alignment, or failure of Eq. (19) is scientifically useful evidence delimiting
the approximation. Smaller batches need not lower every observed `R_T`.

## Paper contribution and positioning

The defensible contribution is a measurement and interpretation result:

> Consecutive-gradient alignment and normalized displacement depend on both
> gradient-estimation noise and the geometry of the visited trajectory. Matched
> sampling controls quantify the former, while full-gradient references along
> SGD trajectories distinguish it from changes induced by parameter motion.

This would strengthen the bridge from Section III's algebra to practical
diagnostics. It would not, on its own, establish a new instability criterion,
improved optimization, or generalization. The inexpensive gradient-only proxy
and the expensive full-gradient/HVP validation must have separate cost claims.

The literature positioning needs to extend beyond Cohen et al.:

- [Cohen et al., Edge of Stability](https://arxiv.org/abs/2103.00065)
  characterize full-GD sharpness and nonmonotone short-term loss dynamics.
- [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/html/1812.06162v1)
  already connect batch size to gradient signal/noise and study its evolution
  during training. The ratio `tr Sigma / ||mu||^2` is not a new contribution.
- [Agarwala and Pennington](https://arxiv.org/abs/2404.19261)
  analyze a stochastic edge and conservative sharpening.
- [Andreyev and Beneventano, v5](https://arxiv.org/html/2412.20553v5)
  study stochastic stability and batch sharpness. Their Appendix G also examines
  pairwise **per-sample model-output gradients** at training stages; these differ
  from consecutive sampled loss gradients along a trajectory. A broad claim that
  prior work ignores gradient alignment or sampling noise would be inaccurate.

Recent related preprints also address the stochastic sharpness gap
([Liao et al.](https://arxiv.org/abs/2604.21016)) and local quadratic/stochastic
stability in LLM training ([Meterez et al.](https://arxiv.org/abs/2607.21716)).
These reinforce the need for a narrow contribution claim. This focused source
check does not establish priority for the entire proposed diagnostic framework.

For later LLM experiments, replace “large batches block noise” with **measured
noise is small relative to the gradient signal at the evaluated states**.
The same nominal batch can have very different `chi_b` over training. Occasional
independent probe estimates can quantify that ratio and its uncertainty.
If Adam is used, raw-gradient coherence and actual update coherence remain
different observables. Large batches alone do not equate them.

## Manuscript organization

Keep III-B as theory, with two targeted prose improvements proposed in the design
file. Keep IV as full-GD evidence. Add a short section immediately afterward,
**Sampling Noise and Displacement under Mini-Batch SGD**, organized as matched
design, frozen-state calibration, moving trajectories, and interpretation limits.
The existing older results would shift accordingly.

Use one compact three-panel figure to test the three equations and a second
figure to compare sampled alignment, full-gradient alignment and displacement
along moving continuations. Put raw curves, full-objective loss histories,
noise-corrected proxy/HVP checks, all seeds and the existing three-model frozen
control in the appendix. Report all continuation loss changes in a compact table
and point to the full histories; do not infer optimization benefit from geometry.

The next decision is whether to adopt this bounded MLP design. The proposal
does not resume the held Analysis020/021 queues, alter their executed inputs,
allocate a new run/analysis ID, or authorize rented compute. Execution estimates
must come from calibration of this specific diagnostic workload.
