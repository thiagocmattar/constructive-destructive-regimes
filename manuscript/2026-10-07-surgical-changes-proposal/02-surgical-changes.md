# Proposed changes to the existing draft

Keep the current LaTeX section structure and reuse verified figures where they
answer the revised question. The surgical unit is a claim, derivation, or
experiment whose role changes. Removing “stability” only from the title would
leave the same unsupported inference throughout the paper.

The targets below refer to the editable archive snapshot in
[draft](../draft/bare_jrnl.tex), using equation labels instead of submitted
equation numbers. The submitted PDF and archive have not been established as
identical revisions. The supplied reviewer reports and
[review map](../../research/REVIEW_CONTEXT.md) provide the concerns, not an
automatic prescription for the rewrite.

## Section changes

| Draft target | Concrete change | Reason and evidence |
| --- | --- | --- |
| [Main file](../draft/bare_jrnl.tex), title, abstract, keywords | Replace the stability promise with temporal interference and trajectory geometry. Remove “Training Stability” from the primary keywords. Name measured raw gradients and optimizer directions. | Resolves the mismatch between the object measured and the outcome claimed. |
| [Introduction](../draft/sections/01-introduction.tex), paragraphs beginning “To narrow this gap” and “The intuition is” | Replace claims about overwriting representations and necessarily slower convergence with a question about accumulation of vectors and loss. | Neither representation retention nor causal learning efficiency was measured. |
| Introduction, contribution list | Promise horizon-resolved characterization, a calibrated measurement procedure, and controlled schedule sensitivity. Reserve a scheduler contribution for successful evaluation. | Separates current findings from the proposed practical extension. |
| [Related work](../draft/sections/02-related-works.tex), section title and concluding paragraph | Broaden to optimization geometry and diagnostics. Keep a short stability context without claiming existing curvature frameworks fail to describe all training dynamics. | The new work complements a different observable rather than replacing a formal stability theory. |
| Related work, “Mini-Batch Stochasticity” | Replace the categorical statement that covariance is not viable online with a distinction between exact full-data covariance and stochastic approximations. | Addresses reviewer C4. The review's Fisher equality requires its model-distribution and regularity assumptions; it is not a blanket equality for empirical data covariance. |
| [Methods](../draft/sections/03-methods.tex), eq:final_theta_expression through eq:thetaT_full | Start with plain mini-batch SGD, derive \(\theta_T-\theta_0=-\eta\sum_t g_t\), and expand its squared norm directly in gradients. Treat Adam separately through its moments and preconditioning. Move the initialization-norm expansion and optional sample-level expansion to an appendix if retained. | Makes the derivation concrete, removes the origin-dependent term, and avoids generic update notation or unmeasured per-sample terms. |
| Methods, “Practical Perspective,” eq:norm_with_correlation and eq:corr | Delete the assumption that interactions beyond lag one are negligible. Define \(I_k\), \(J\), the measured residual, and \(\rho_t\) as the lag-one measurement. | The existing temporal evidence contradicts the approximation as a cumulative proxy. |
| Methods, eq:rho and paragraph after fig:traj_decomp | Retain the cosine formula with defined zero norms and update indexing. Replace “directly governs \(S_3\)” with “measures adjacent alignment.” | Adjacent cosine and the all-pair interaction can have opposite signs. |
| Methods, new short subsection | Define \(A,B,R\), raw/clipped/momentum/adaptive/actual-update vectors, and fixed CountSketch history with calibration. | Supplies the measurement method missing from the draft and makes the Adam comparison interpretable. |
| Methods, “Effects on Loss Dynamics,” and [main-file appendix](../draft/bare_jrnl.tex), appendix:two_step | Keep the expansion only as a qualified local illustration, preferably in the appendix. Audit the remainder and noise/curvature terms. Remove conclusions about total loss change drawn from the cosine term alone. | The LR intervention shows why sign alone cannot carry the learning claim. |
| [Results](../draft/sections/04-results.tex), opening and “Trajectory Regimes in Plain SGD” | Rename around temporal interference. Label every “constructive/destructive” statement by vector and horizon. Describe the old CIFAR phases as findings in that study, not a universal three-phase sequence. | LLM late phases and long-lag backgrounds do not fit the original universal narrative. |
| Results, “Momentum as Temporal Averaging” | Preserve the temporal-filter intuition. Distinguish the unnormalized SGD velocity \(p_t=\mu p_{t-1}+g_t\) from Adam's EMA \(m_t=\beta_1m_{t-1}+(1-\beta_1)c_t\). Add the proposed preconditioning-free comparison when measured. | The two recursions have different amplitude normalization; the new experiment must isolate the intended effect. |
| Results, “Weight Decay as Recency Bias” | Retain its stated SGD recurrence in a shortened subsection or appendix; qualify its interpretation and the requirement \(0\leq1-\eta\lambda<1\). | Its recency weighting is not a derivation for all AdamW update components or a general robustness guarantee. |
| Results, “Stability Across Learning Rates and Batch Sizes” | Rename as hyperparameter effects on geometry and loss. Audit whether each reported \(S_3\) is full, truncated, raw-gradient, or velocity-based before reusing it. | Existing figure PDFs alone do not establish that the plotted estimator matches the revised definition. |
| Results, “Batch Representativeness” and eq:energy-sample | Make this secondary to the new core. If retained, audit the estimator and plotted square root; preserve the original source and regenerate any corrected figure from evidence. | An unbiased U-statistic estimate of energy distance can be negative. Moving it to an appendix does not fix the issue. |
| [Conclusion](../draft/sections/05-conclusion.tex) | Restate the measured distinction between local, cumulative, and loss behavior. Discuss scheduling as tested sensitivity plus a conditional research direction. | Prevents the abstract and conclusion from promising more than the experiments show. |

The energy-distance repair, if that experiment remains, should choose explicitly
between reporting a signed unbiased estimate without taking its square root,
or recomputing the nonnegative empirical V-statistic for Euclidean distance.
Changing denominators or clipping negative values silently would change the
estimator and is not an editorial correction.

Also repair the indexing and units when replacing the practical equations:
under the proposed \(t=1,\ldots,T\) convention, adjacent sums begin at \(t=2\).
If the sample-level \(S_1,S_2,S_3\) notation is retained, carry its
\(\eta^2/m^2\) scaling consistently rather than labeling scaled batch-level
terms with an unscaled sample-level symbol.

## Proposed replacement passages

These are candidate text blocks for review, not edits already applied to TeX.

### Abstract based on completed evidence

> Sequential gradients can oppose one another locally while reinforcing one
> another over longer horizons. We study this temporal geometry through an
> exact decomposition of SGD displacement in terms of mini-batch gradients,
> adjacent cosine similarity, lag-resolved interactions, and a cumulative
> coherence ratio. We treat Adam's adaptive directions separately.
> In Pythia/MiniPile experiments, negative adjacent raw-gradient alignment
> coexists with positive total cross-step interaction, making lag-one
> truncation an inadequate description of cumulative geometry. Paired
> measurements also distinguish raw gradients from Adam's adaptive directions.
> Fixed random sketches, calibrated against exact inner products and
> independently accumulated totals, permit temporal analysis without storing
> full gradient histories. In checkpoint-matched learning-rate interventions,
> faster decay reduces negative alignment, while held learning rates achieve
> lower validation loss at the measured endpoint. These observations establish
> the importance of specifying both vector and horizon when interpreting
> interference. They motivate evaluating geometry as an input to step-size
> selection, with loss measured independently rather than inferred from
> alignment.

This version centers the new LLM evidence. If the CIFAR study remains a main
result, add one bounded sentence after auditing its diagnostics. Do not add
momentum-only or scheduler-performance claims before those tests exist.

### Introduction and contribution statement

> Under mini-batch SGD, the displacement from initialization is the sum of
> gradient steps. Its magnitude depends on both the gradient magnitudes and
> their relative directions. We first study these interactions across temporal
> lags, then examine how Adam's temporal averaging and preconditioning change
> them. Finally, we test their response to learning-rate interventions.
> Adjacent cancellation is a geometric observation; whether it accompanies
> useful progress is a separate empirical question.

Recommended contributions:

1. An exact SGD displacement decomposition that distinguishes adjacent
   gradient alignment, lag-resolved interaction, and cumulative coherence,
   followed by explicit definitions for Adam's moments and adaptive directions.
2. A calibrated procedure for measuring temporal gradient geometry, with exact
   aggregate checks and a clear separation between projection sensitivity
   and variation across training seeds.
3. Empirical characterization across the measured model sizes and seeds,
   together with checkpoint-matched schedule interventions demonstrating
   that alignment and loss need not improve together.

Optimizer-component attribution and an effective feedback rule can strengthen
items 2 and 3 after their dedicated tests. The elementary norm identity itself
should not be advertised as the main theoretical novelty.

### Lag one and loss interpretation

> We retain adjacent cosine as an inexpensive local diagnostic. We do not
> assume that it summarizes all cross-step interactions. Instead, lag profiles
> and exact streaming aggregates quantify the omitted terms. The resulting
> measurements describe reinforcement and cancellation in the chosen vector
> space; they do not alone determine loss change, forgetting, or stability.

### Practical direction

> Learning-rate interventions alter local gradient geometry, but reducing
> cancellation need not improve the loss reached at a fixed token budget.
> A useful controller must therefore learn or validate when a geometric
> response predicts a beneficial intervention, rather than treating positive
> alignment as an objective in itself.

## Figure sequence

| Proposed role | Figure or source | Revision needed |
| --- | --- | --- |
| Intuition before LLM results | New quadratic/Rosenbrock trajectories with objective values | Show oscillatory descent and failure cases using exact gradients and updates. Planned only. |
| Local versus cumulative geometry | [Lag profiles](../../analyses/005-2026-10-04-temporal-lag-profiles/figures/lag_profiles.pdf), [fixed origins](../../analyses/005-2026-10-04-temporal-lag-profiles/figures/lag_profiles_fixed_origins.pdf), and [all-lag accumulation](../../analyses/005-2026-10-04-temporal-lag-profiles/figures/cumulative_lag_interaction_all_lags.pdf) | Make the primary profile and exact cumulative endpoint central; fixed origins is the coverage control. A zoom at lags 1–8 would make the rebound visible. |
| Optimizer transformation | Existing [raw/adaptive figure](../../runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/figures/raw_adaptive_comparison.pdf) plus planned \(g,c,m,\hat m\) measurements | Place raw and first-moment \(A,B\) on shared axes; use separate panels for preconditioned scales. Never splice different trajectories. |
| Practical intervention and loss | [Schedule and validation](../../analyses/004-2026-10-04-lr-schedule-intervention/figures/schedule_and_validation.pdf), with [14M](../../analyses/004-2026-10-04-lr-schedule-intervention/figures/loss_and_raw_cosine_14m.pdf) and [70M](../../analyses/004-2026-10-04-lr-schedule-intervention/figures/loss_and_raw_cosine_70m.pdf) | Keep the held-LR/lower-loss result visible and add replication when available. |

Each new or recomposed publication figure should be generated in its owning
run or analysis, with a local observation recording sources, reduction,
coverage, uncertainty, caption, and script. Link the result into the manuscript;
do not treat this planning folder as an experiment-artifact store.

## Review completion criteria

Before applying the rewrite, reconcile any cited submitted equation with the
archive snapshot, audit retained CIFAR estimators, and complete the focused
novelty comparison. After applying it, check all abstract/introduction/conclusion
claims against the evidence map, validate equation indexing and vector labels,
and compile and inspect the manuscript. Preserve the original archive, reviews,
figures, and supplied reflections throughout.
