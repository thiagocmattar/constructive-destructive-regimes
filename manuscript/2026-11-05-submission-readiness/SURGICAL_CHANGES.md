# Surgical changes proposed for the current manuscript

Apply these changes to the snapshot recorded in [source_inventory.json](source_inventory.json). They are a proposal, not an edited manuscript. Preserve the training results and the author's constructive/destructive framing while making its geometric meaning and empirical scope explicit.

S01–S06 address the core reviewer objections and the missing validation disclosure. S07–S12 align the experiments, figures, and closing claims. None requires new training. New measurements for stronger claims are separated at the end.

## S01 Align the title and opening promise

**Anchor:** `bare_jrnl.tex`, title; `01-introduction.tex`, first two paragraphs. **Objective:** prevent R1 C1/C2 from recurring before the reader reaches the qualifications. **Priority:** essential.

Proposed title:

> Constructive and Destructive Training Regimes: Gradient Reversal, Displacement, and Loss Dynamics

Replace the first paragraph's final sentence about why the listed heuristics work with:

> These practices motivate a measurement question: what do the gradients already computed during training reveal about local reversal, accumulated parameter motion, and changes in the objective?

Replace “Training becomes unstable when updates are too large…” with:

> On a fixed positive-definite quadratic, sufficiently large steps amplify perturbations along sharp directions. Neural-network trajectories can instead change their geometry while moving, allowing local reversals to coexist with sustained loss reduction.

Keep the existing scaling motivation, but shorten it to avoid implying that the experiments establish large-model pretraining performance. **Completion check:** title, abstract, introduction, and conclusion name the same observable contribution; none equates oscillation with divergence or slow convergence.

## S02 Define exactly what constructive and destructive mean

**Anchor:** `03-methods.tex`, paragraph beginning “Constructive and destructive behavior…”; `01-introduction.tex`, paragraph beginning “Destructive alignment alone…”. **Objective:** distinguish a local geometric label from stability and whole-path interaction. **Priority:** essential.

Replace the two sentences defining persistent regimes with:

> For nonzero consecutive gradients, we call a transition locally constructive when rho_t > 0 and locally destructive when rho_t < 0. These labels describe reinforcement or reversal of consecutive plain-SGD displacements. We describe sustained behavior through the explicitly stated temporal summaries in the experiments; the local label does not determine the sign of the all-pair interaction or the loss change.

Use TeX notation when applying. Add one sentence at the end of II-A:

> We use stability here for the contraction property of this fixed quadratic map; the trajectory diagnostics below do not constitute a general stability or convergence guarantee.

In I, change “an SGD step” in the joint reversal/growth sentence to “a full-batch GD step,” and retain “within a local quadratic approximation.” **Completion check:** no undefined persistence threshold is implied; stochastic loss statements retain their noise term; positive local sign is never called useful learning.

## S03 State the contribution relative to prior work

**Anchor:** end of `02-related-works.tex`, before III; contribution list in `01-introduction.tex`. **Objective:** answer R2 (1)/(2) directly. **Priority:** essential.

Insert a compact related-work paragraph:

> High-probability analyses of stochastic optimization address optimization and population-risk guarantees under explicit assumptions on smoothness and noise [Li and Liu; Lei]. Our trajectory identities provide geometric accounting conditional on a realized path, rather than a competing bound on iterate radius or generalization. Consecutive-gradient tests also precede this work [Chee and Toulis; Pesme et al.], and two-point secant methods and hypergradient adaptation already extract information from consecutive gradients [Barzilai and Borwein; Baydin et al.]. Here we use these measurements to distinguish local reversal from all-lag accumulation and objective progress, and to identify how sampling and Adam memory alter their interpretation.

Cite the primary sources linked in the review. Add Lei and Li–Liu to the bibliography; the other four entries already exist. Keep their characterizations narrow rather than asserting that each proves the same iterate-bound theorem.

In III-B, after Eq. (15), add the identity `kappa_hat_t = s^T y / (s^T s)` for `s = −eta*g_(t−1)` and `y = g_t−g_(t−1)`, citing Barzilai–Borwein. Rewrite contribution 2 as:

> A unified interpretation of the consecutive-gradient secant under full-batch GD, mini-batch sampling, and Adam, with explicit finite-step and optimizer-memory corrections.

**Completion check:** all six works appear in the compiled bibliography; the scalar secant and cosine are not claimed as inventions; the new empirical distinction is stated positively.

## S04 Close the covariance objection explicitly

**Anchor:** `02-related-works.tex`, after “Exact evaluation requires per-example information absent from the aggregated training gradient.” **Objective:** finish R1 C4 without adopting an unjustified Fisher equivalence. **Priority:** small essential edit.

Insert:

> This does not preclude stochastic estimates of covariance summaries or Fisher-type approximations; our comparison concerns exact empirical covariance evaluation, rather than the feasibility of online approximation in general.

Do not add a full-dataset empirical-covariance-equals-Fisher equation. **Completion check:** no claim says online covariance monitoring is impossible or necessarily requires a full pass at every step.

## S05 Clarify availability and smoothness at first use

**Anchor:** `03-methods.tex`, immediately after Eq. (15) and its path-average interpretation. **Objective:** separate online availability from prospective prediction, and smooth theory from nonsmooth experiments. **Priority:** essential.

Insert:

> The statistic becomes available when g_t is evaluated, after the preceding parameter update. It characterizes that completed step and is not, by itself, a forecast of the next loss change.

After the smooth path-average interpretation, insert:

> This Hessian interpretation assumes smoothness along the complete step. For a step crossing a nonsmooth activation boundary, the endpoint secant remains defined, but the classical Hessian integral need not capture the gradient change.

Replace “conditions for loss decrease and spike formation” with “local quadratic approximation to loss change” in III-E's heading. Replace its final two sentences with:

> The criterion characterizes a single completed GD step under the stated approximation. The Adam excursions below require a separate treatment because their updates follow transformed gradients.

**Completion check:** no unqualified transfer of the threshold two to Adam; no retrospective quantity described as a validated alarm.

## S06 Restore the retained one-step approximation audit

**Anchor:** `04-gd-experiments.tex`, after the paragraph ending “without additional gradient evaluations or Hessian-vector products.” **Objective:** confront the main theory with the directly relevant empirical measurement. **Priority:** highest scientific edit using existing data.

Insert a short paragraph:

> We also compare the unsmoothed endpoint-gradient approximation in Eq. (23) with the observed loss change over the same completed step. Across 2,999 eligible transitions per condition, the sign-disagreement fraction ranges from zero to 44.2%, reaching its maximum for the Transformer at eta = 0.2. In that condition, excluding changes within 10^−5 times the larger endpoint loss gives 43.7%. These retrospective discrepancies limit a stepwise loss-sign interpretation on the nonlinear trajectories; they do not contradict the exact displacement identity or the observed longer-window loss reduction.

Add a compact Appendix B table containing the nine counts from Analysis024, explicitly reporting denominator 2,999 and the endpoint indexing. Include the recorded sensitivity convention and explain that it is not a numerical-error bound. Trace the table to `section_iv_review_checks.json` and Observation007; verify source hashes and numerical reductions before integration. Do not substitute the existing two-update loss figure for this test.

**Completion check:** all nine rows match retained evidence; the 44.2% and 43.7% statements have their different denominators; no binomial confidence interval or prospective prediction claim is added.

## S07 Make the GD result the organizing empirical finding

**Anchor:** `04-gd-experiments.tex`, opening and IV-B. **Objective:** answer “what do we learn beyond EoS?” using the strongest existing result. **Priority:** important.

Add this sentence after the opening paragraph:

> The central distinction is that seven trajectories have negative cumulative adjacent interaction, while all nine have positive interaction when every temporal pair is included.

Keep the existing C/D and zero-cross-term numbers, but consolidate their repeated interpretation into one paragraph. Keep the loss measurements separate: C/D accounts for motion, while loss establishes objective progress. Correct “range … from 1.97 and 1.99” to “range … from 1.97 to 1.99.”

**Completion check:** the first empirical page gives the all-lag result before lengthy spectral context; single-initialization scope remains explicit; no equation or observation is lost.

## S08 Give the SGD experiment a question that it actually tests

**Anchor:** `05-sgd-noise-experiments.tex`, start of the subsection. **Objective:** stop readers treating moving continuations as a validation of the fixed-state cosine approximation. **Priority:** important, small edit.

Insert:

> We next ask how batch size changes the alignment observed along evolving training trajectories. This comparison includes both sampling effects and changes in the states visited; it does not isolate the fixed-state approximation in Eq. (18).

Retain the existing model-specific results, three-block range, ResNet group sampling, unequal exposure, and loss counterexample. Move the longer proposed frozen-two-endpoint control to the closing limitations paragraph rather than interrupting the results.

**Completion check:** every interpretation of Fig. 4 refers to observed sampled gradients along moving paths; no new experiment or assumed attenuation mechanism is introduced.

## S09 Reorder and shorten the Adam argument

**Anchor:** `06-adam-diagnostics.tex`, opening, “Current slope versus optimizer direction,” and V-E. **Objective:** connect Adam to the main paper before introducing detailed event statistics. **Priority:** important.

Add at the opening:

> Adam provides a test of the interpretation's limits: raw-gradient reversal no longer implies reversal of the applied update. We first distinguish those quantities, then describe loss-defined excursions and the corrections needed to interpret the gradient response as curvature.

Move the short current-slope paragraph to immediately after the qualifying-event population is defined, so the 794/794 result precedes severity regressions. Change “locally descending steps” to “updates with a descending first-order contribution”; retain the explicit fact that the realized loss rises over the event.

Move the full-event density display (Fig. 7) and the original product-fit display (Fig. 9) to supplementary material, retaining one sentence of their results and their source captions. Keep event boundaries, overview, examples, threshold/censoring counts, and norm-only comparison available. Before V-E insert:

> The event measurements show what the response accompanies. The following identity addresses a different question: which memory and finite-step terms separate it from a preconditioned-curvature proxy?

**Completion check:** event anatomy, approximation identity, and causal explanation are not conflated; the original evidence remains preserved; the narrative reaches its central result sooner.

## S10 Align the Adam figures with the definitions

**Anchor:** Fig. 8 generator/asset and caption; Fig. 10 and its supporting Analysis057 reductions. **Objective:** remove competing notation/reference conventions and visualize the strongest comparison. **Priority:** important presentation fix.

For Fig. 8, relabel the response axis z_(s+1), retain the same selected events, mark a_i/p_i/c_i, and show L(a_i). If the compact window excludes a_i, extend it or visibly label the off-panel event start. Either shade the formal rise interval or label the +10%-crossing band as a separate late-rise interval. Do not quietly change the detector or selected population. Existing first-state norm normalization may remain if clearly labeled.

For Fig. 10, replace or supplement within-rate R² with the seven held-out-rate squared-error comparisons for log A versus log A + rho. Keep sample counts and the shared-seed limitation. Retain all in-sample fits in supplementary evidence. No new regressions are needed if the retained Analysis057 outputs supply the required values.

**Completion check:** regenerate with owning analysis scripts and observation records; verify unchanged source event selection and plotted arrays except intentional labels/reference transformations; inspect at final column width. Ensure the adverse held-out result is as visible as the attractive in-sample correlations.

## S11 Make abstract and conclusion selective

**Anchor:** `bare_jrnl.tex`, abstract; `05-conclusion.tex`. **Objective:** present a coherent contribution rather than an inventory of every analysis. **Priority:** final editorial pass.

Organize the abstract around four statements: the local/cumulative distinction; the conditional secant interpretation; the GD/SGD empirical finding; and the Adam scope result. A proposed closing sentence is:

> Under full-batch Adam, reversal also occurs during ordinary training, and adding alignment to norm-based severity models does not improve transfer consistently across rates; an analytical memory correction specifies what is required for a curvature interpretation.

Keep the conclusion's currently accurate negative findings. Condense the list of future studies into one sentence tied to the claim it would enable. Avoid ending the abstract with several uncompleted projects or presenting the memory identity as measured neural fidelity.

**Completion check:** every abstract sentence maps to a theorem/identity or displayed result; no practical benefit is inferred merely from computability.

## S12 Complete a source and layout pass

**Anchor:** active TeX input graph and final PDF. **Objective:** remove residual friction after the scientific edits. **Priority:** final verification.

- Use z consistently for raw Adam response, retaining eta*kappa_hat only for the explicitly related GD statistic.
- Keep state s and gradient index t distinguished in captions; do not interpret the separately recomputed state-3256 lag profiles as part of the original 3,000-step history.
- Enlarge or simplify Table I; move detailed phase summaries to supplementary material if necessary rather than shrinking type further.
- Remove unused draft-change machinery only after checking whether it is active; keep supplied originals and inactive historical assets intact.
- Check every figure/table/reference after recompilation, especially appendix tables whose first mentions precede their placement.
- Add a concise reproducibility statement linking the experimental specification/code when a shareable destination is approved. Do not invent a public repository or publish local records.

**Completion check:** clean build with resolved references/citations, no material overfull boxes, full visual review, source-to-PDF consistency, and a reviewed diff limited to the authorized revision. Settle venue formatting only after the venue is selected.

## Evidence needed only for stronger claims

| Stronger claim | Minimum useful next evidence | Why it is separate from this edit |
| --- | --- | --- |
| Secant reliably approximates pointwise curvature | Matched same-objective directional HVPs and finite-step discrepancies at declared smooth states; retain failures | Subset maximal spectra do not test this. |
| Adam response approximates the cited one-HVP proxy | Measure both corrections in the existing identity on matched checkpoints, including transient memory | The current bridge is analytical and explicitly uncalibrated. |
| Alignment contributes transferable spike information | Independent initialization blocks, norm/loss-only baselines, temporally valid predictors, predefined future outcomes | Retrospective peak norms and same-seed rate holdouts do not establish prediction. |
| Sampling attenuates negative alignment at fixed geometry | Independent sampling at two frozen endpoints with known full-gradient cosine | Moving continuations cannot isolate measurement attenuation. |
| Monitoring is useful and cheaper in realistic pretraining | Stochastic Adam on matched objectives with measured wall time, memory, fidelity, and a one-HVP comparator | Zero extra backpropagations does not itself demonstrate operational savings. |

These are options for the author's scientific direction, not an authorized launch plan or a condition for making the existing descriptive paper coherent. First complete the small edits and the missing evidence disclosure; then decide whether the intended contribution requires expanding the experiments.
