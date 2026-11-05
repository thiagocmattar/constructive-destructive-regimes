# Review of the current manuscript against the previous submission

The current paper has substantially addressed the original mathematical objections. Its strongest contribution is now a precise distinction between local gradient reversal, cumulative displacement, and objective progress. It is not yet consistently presented as that paper: the title still promises SGD stability, several passages imply a practical curvature diagnostic more strongly than the validation supports, and the Adam section carries a second, partly unvalidated contribution.

The recommended next step is a bounded editorial revision using existing evidence, especially the retained one-step loss audit. A new experiment campaign is not a prerequisite for correcting the present argument. Stronger claims about reliable monitoring, spike prediction, causal mechanisms, or computational advantage would require additional evidence.

See [the surgical changes proposal](SURGICAL_CHANGES.md) for exact anchors, replacement text, objectives, and completion checks. No proposed manuscript change has been applied.

## Scope and source identity

Reviewed on 5 November 2026: the actual 18-page [current PDF](../draft/bare_jrnl.pdf), its recursively compiled TeX inputs, figures and tables, and the four files in [previous-submission](../draft/previous-submission/). The submitted manuscript is 12 pages. Source fingerprints and active-input line anchors are in [source inventory](source_inventory.json). Page numbers below refer to the current PDF unless explicitly marked submitted.

The current TeX/PDF includes the paired-motion and Adam-memory material associated with Analyses056/057. Historical README statements about restoration or earlier page counts are not a reliable description of this snapshot. Inactive `04-results.tex` and `06-appendix.tex` are not treated as current claims.

The supplied decision was from **IEEE Transactions on Neural Networks and Learning Systems**, manuscript TNNLS-2026-P-48775, dated 8 September 2026. The letter says not to submit a revised version to that journal. This is preparation for a new submission elsewhere, not an invited TNNLS revision. The advisors mention TPAMI as a possibility, not a settled destination; this review does not choose a venue or assume its page limit.

The image-only `rejection_email.pdf` was read visually, including its advisor discussion. Frederico summarizes the central scientific concerns: define stability, justify the parameter-norm connection, support assumptions empirically, moderate mini-batch diagnostic claims, and improve comparisons. Antonio encourages using the reviews to improve and submit elsewhere; Cristiano agrees with prioritizing corrections. The remaining messages concern timing, staffing, and acknowledgments rather than additional technical requirements.

## Disposition of every reviewer concern

| Concern | Status in this snapshot | Evidence and remaining action |
| --- | --- | --- |
| R1 C1: define stability and distinguish it from interference | **Partially addressed** | II-A now correctly restricts contraction to a positive-definite fixed quadratic; III distinguishes cancellation from forgetting and objective progress. But the unchanged title and opening still suggest a broader stability framework. Define the paper's scope explicitly and identify constructive/destructive as local geometric labels. S01–S02. |
| R1 C2: justify parameter norm as a stability measure | **Addressed by reformulation, not by proving the original claim** | III uses displacement from initialization, retains all temporal interactions, defines path-relative motion, and explicitly rejects interpreting it as retained learning. IV shows low path ratios with loss reduction, and positive all-pair interaction despite negative adjacent interaction. Do not describe this as validation of the old parameter-norm argument. S02–S03. |
| R1 C3 and R2 (3): unjustified neglect of nonadjacent interactions | **Addressed** | Eqs. (7)/(9) retain all pairs. IV-B reports positive total interaction in all nine runs; IV-C/Fig. 3 measures lags through 256 at three sampled endpoints. The old approximation has been removed, and the new evidence demonstrates why it should not be restored. Finite lag coverage is sufficient for this disposition because the theory no longer assumes lag truncation. |
| R1 C4: covariance cannot be monitored online is too strong | **Largely addressed; explicit acknowledgment still useful** | II-B now restricts the expensive quantity to exact per-example covariance and no longer rules out online estimation. Add one sentence acknowledging stochastic approximations. The reviewer's model-distribution Fisher identity must not be equated automatically with empirical covariance on observed labels. S04. |
| R2 (1): no explicit iterate bound or comparison with norm-control literature | **Partially addressed through changed scope; literature gap remains** | Current theory makes geometric identities and conditional local expansions, not a high-probability radius bound. That is legitimate if stated plainly. Lei and Li–Liu are still absent from the compiled references. Add their distinct questions and explain why this paper does not supply a competing convergence/generalization bound. S03. |
| R2 (2): squared-norm expansion is standard and insufficiently novel | **Still material** | Pairing, zero-cross-term references, all-lag measurements, and the optimizer-memory discrepancy add substance. Their identities are nevertheless elementary accounting, and the secant statistic has close classical precedents. The claimed contribution should be the integrated characterization and empirically demonstrated distinctions, with explicit prior-art positioning. A larger equation count alone does not resolve this objection. S03, S06, S09. |
| R2 (4): submitted Eq. (30) can yield negative estimated batch distance | **Removed from the current scientific argument** | The old input-space batch-representativeness experiment and its estimator are not compiled. Current Eq. (30) is an unrelated Adam descriptor. The old formula used off-diagonal within-sample terms with denominator m(m−1), so negativity of a finite-sample unbiased estimator is possible; this does not make population energy distance negative. Record removal, not a mathematical repair or a verdict that all energy-distance estimators are invalid. |
| Editor: connection to stability, missing related work, restrictive assumptions, insufficient value | **Mixed** | The restrictive assumption and norm interpretation have substantially changed. Contribution framing, prior-art comparisons, and validation of the curvature/loss diagnostic remain the highest priorities. |

The paper therefore answers several objections by replacing the original claim. That is a stronger and more honest response than attempting to defend the submitted interpretation of cancellation as overwritten learning.

## Current inconsistencies and priorities

### 1. The promised object is broader than the demonstrated object

The title is still “A Framework for Interpreting SGD Stability.” The abstract, IV, and conclusion are much more careful: they distinguish motion from loss and treat Adam associations as descriptive. In I, however, the introduction begins with large-model stability and unexplained heuristics, and says joint reversal and growth determine whether “an SGD step” decreases loss under a local quadratic approximation. III-E's simple sign criterion is specifically full-batch; stochastic SGD adds a noise term. The opening therefore compresses away a qualification that the theory later needs.

The defensible central statement is: **consecutive reversal can coexist with positive cumulative interaction and decreasing loss; the same raw-gradient response has different curvature interpretations under GD, sampling, and Adam.** This preserves the author's training-dynamics motivation while giving the reader a single question that all sections answer.

### 2. A direct test of the central loss approximation exists but is missing

III-E gives a quadratic-model loss-sign criterion. IV shows smoothed secants near two, spectral context, and two-update loss changes, but none is a direct fidelity test of the one-step formula. Fig. 11 asks whether paired updates descend; it does not validate Eq. (23)/(24).

The retained [Analysis024 observation](../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md) and [audit JSON](../../analyses/024-2026-10-17-eos-gd-lr-calibration/section_iv_review_checks.json) already compare all 2,999 eligible unsmoothed transitions per condition. Opposite-sign counts are 0/145/281 for the MLP's increasing rates, 595/694/814 for ResNet, and 0/542/1,325 for the Transformer. The last is 44.2%; excluding changes within the recorded 1e−5 relative threshold leaves 1,239/2,837, or 43.7%.

These are retrospective comparisons, not forecast accuracy or independent trials. They do not isolate numerical error, nonsmoothness, and finite-step approximation error. They do establish that the loss-sign criterion should not be presented as empirically reliable throughout these trajectories. Restore a compact summary and its sensitivity caveat. This is the highest-value existing-evidence change.

### 3. Prior art is missing exactly where novelty is vulnerable

The compiled bibliography has EoS and Bai, but not the already-stored Barzilai–Borwein, Chee–Toulis, Pesme et al., or Baydin et al. entries. Consecutive-gradient information and inexpensive gradient-based adaptation are established ideas. The proposed scalar has the secant form

`kappa_hat = (sᵀy)/(sᵀs)`, with `s = −eta*g_previous` and `y = g_current − g_previous`.

This identity follows directly from the draft's definition and links it to classical two-point curvature estimates. It should be acknowledged, while explaining the distinct use here: measuring local reversal, its relationship to path accumulation, and the corrections required under sampling and Adam. Do not claim the squared-norm expansion, the cosine itself, or Hessian-free scalar curvature estimation as unprecedented.

Primary sources checked for this review: [Barzilai and Borwein](https://doi.org/10.1093/imanum/8.1.141), [Chee and Toulis](https://proceedings.mlr.press/v84/chee18a.html), [Pesme et al.](https://proceedings.mlr.press/v119/pesme20a.html), and [Baydin et al.](https://arxiv.org/abs/1703.04782). Pesme et al.'s limitations of a consecutive-gradient convergence test are particularly relevant to the paper's scope. For the reviewers' requested comparison, [Lei's author-hosted paper](https://leiyw.github.io/paper/TPAMI2025.pdf) studies generalization via gradient uniform convergence, while [Li and Liu](https://proceedings.mlr.press/v162/li22q.html) give high-probability optimization/generalization guarantees under heavy-tail conditions. A precise contrast is more useful than grouping all of these as generic “stability” work.

### 4. The temporal and smoothness qualifications need consistent placement

The secant at gradient index t uses g_t, which is evaluated after the step whose change is being explained. It is available during training, but does not predict that already-completed step. The distinction should appear at first definition rather than be left for the reader to reconstruct from indices.

The GD path-average Hessian identity assumes smoothness along the whole step. ResNet uses nonsmooth activations; the Adam section explicitly acknowledges activation-boundary limitations, whereas the GD interpretation relies mainly on a generic “smooth” qualification. Across boundaries, endpoint gradients and their secant are still measurable, but a classical Hessian integral need not account for gradient jumps. Neither a sparse subset spectrum nor near-antiparallel gradients validates that integral for every experimental step.

“Persistent” sign defines regimes in III without a declared window or occupancy rule. The figures use trailing means, which can delay sign changes; a mean sign is not the same thing as every step having that sign. No new threshold is needed: define stepwise labels and describe sustained behavior using the reported window summaries.

### 5. The Adam section has a useful result, but competing emphases

V-A–D carefully documents loss-defined events, includes censored candidates, controls for ordinary oscillation, and compares alignment with norm-only descriptors. This is a substantial improvement. Particularly useful are the 794 rise-phase outgoing updates with negative gradient-dot-increment and the failure of added cosine to improve held-out-rate squared error in all seven log-norm folds.

Yet the visuals emphasize the original product correlation and within-trajectory R². The stronger limitation—lack of transfer—is only prose. Six main-text Adam figures and a large table make an exploratory subproblem dominate a paper introduced through SGD geometry. V-E then introduces the analytical memory bridge after a long event-description detour.

Keep the empirical characterization, but give the slope/update distinction and norm-only comparison more prominence. Reduce the density and product-fit displays to supplementary material, preserve threshold/censoring accounting, and place the purpose of the analytic bridge before its equations. The identity is a meaningful connection; it is not yet a neural-network validation or a causal explanation of spike formation. [Bai et al.](https://proceedings.mlr.press/v306/bai26o.html) already present a mechanism and a directional-curvature predictor, so the incremental contribution must be explicit.

### 6. Figure semantics do not fully match the revised prose

Fig. 8 on p. 13 still labels the raw Adam response with scaled-curvature notation, while V-A deliberately names it z. Its example windows also use a first-displayed-state reference, and shaded bands begin at a +10% crossing; the formal event begins at a_i and is referenced to L(a_i). The caption discloses this, so the plot is not secretly incorrect, but readers must reconcile two definitions of the illustrated episode.

Use z on Adam axes and mark the actual a_i, p_i, c_i and fixed reference in the examples. Preserve the same selected events and measurements. Fig. 10 emphasizes more flexible in-sample fits; a compact held-out-rate error comparison would communicate the actual conclusion better. Table I is dense and small at publication size. These are targeted presentation changes, not reasons to regenerate the experimental program.

### 7. Scope limitations are mostly disclosed, but should guide the framing

The full-batch comparisons use one initialization per model, calibrated rates, fixed objectives, no augmentation/dropout, and hardware differences in some arms. The SGD study uses three blocks and dependent repetitions, with ResNet groups rather than arbitrary image mini-batches and unequal exposure at equal updates. Its claims are already appropriately limited. The same-state noise approximation cannot explain attenuation of a negative cosine between changing endpoints, and IV-D now says so correctly.

These controlled studies support mechanistic measurement questions. They do not establish generality to contemporary large-scale stochastic Adam pretraining, nor monitoring savings. Independent seeds and direct same-objective curvature checks would strengthen a later version, but the immediate task is to stop the introduction from implicitly promising those results.

## What should be preserved

- The all-lag identity and the explicit distinction between local and total interaction.
- The zero-cross-term reference and paired C/D interpretation, including boundary treatment and undefined-denominator conventions.
- The finding that all nine endpoint total interactions are positive despite seven negative adjacent-interaction totals.
- The moving-trajectory qualification in the SGD experiment and its model-specific findings.
- Adam event definitions, censored incidence, the norm-only comparisons, and the negative transfer results.
- The raw-gradient/actual-update distinction and conditional optimizer-memory identity.

Rewriting the paper as a claim that constructive alignment is always good would discard its strongest evidence.

## Verification and readiness

The review checked the active input graph, local labels/citations/assets, numerical statements against the cited retained summaries, and the 18-page PDF's visual organization. All pages were rendered; key equations, examples, and tables were inspected in context. There are 11 figures and three tables. No obvious clipping or unresolved-reference marker was visible. Dense type in Table I and the competing Fig. 8 references remain presentation concerns. This was not a fresh LaTeX build or a rerun of scientific experiments; the existing PDF was the requested review target.

A narrow submission claim can be made substantially more coherent through S01–S12. Before calling the result submission-ready, verify the revised PDF, restore the direct approximation audit, complete the prior-art contrast, and settle the destination's actual format requirements. If the paper continues to promise validated online stability monitoring, a predictive alarm, or superiority over the one-HVP baseline, editing alone is insufficient.

No experiment, cloud execution, publication, or push is part of this review. No new hypothesis/run/analysis ID is allocated because this is an editorial assessment of existing evidence. Supplied PDFs and manuscript sources remain unchanged.
