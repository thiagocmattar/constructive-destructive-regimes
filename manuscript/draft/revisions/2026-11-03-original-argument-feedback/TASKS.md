# Author-feedback checklist

Baseline: the original-argument restoration. All actions below map to the
41 annotations in [feedback.md](feedback.md); the supplied PDF remains intact.
Work proceeds in the order below. Keep the introduction, Adam section and
conclusion unchanged. Restrict additional edits to necessary notation,
references, Figure 1 and page layout.

## 1. Extract and assess before editing

- [x] Extract all 41 annotations with page, type, selected passage and source hash.
- [x] Preserve the original feedback, preceding review and unannotated sections.
- [x] Check the break-even/EoS literature and the distinction between local instability and longer-term descent.
- [x] Check practitioner reports and the gradient-interference connection to forgetting.
- [x] Inspect the existing spectral measurements, coverage and plotting clocks for Figure 1.

## 2. Stability background (C01-C11)

- [x] C01: Use the author's intuitive opening, with reinforcement, reversal and the curvature/sampling mechanisms.
- [x] C02: Remove "Background" from the Section II title.
- [x] C03: Consolidate repeated curvature citations and prevent isolated citation lines.
- [x] C04: Remove the one-use perturbation notation; explain eigencomponent contraction directly.
- [x] C05: Strengthen the computational reason for an online diagnostic without Hessian evaluation.
- [x] C06: Keep the noise-estimation cost only where it supports that diagnostic argument.
- [x] C07: Critically assess break-even relevance; retain its useful connection without the lengthy auxiliary equation/notation.
- [x] C08: Explain EoS coexistence of oscillations and descent, and relevant subsequent explanations.
- [x] C09: Make the transition from expensive geometric measurements to training-gradient diagnostics explicit.
- [x] C10: Consolidate that transition at the end of Section II instead of repeating it.
- [x] C11: Shorten practical-pretraining discussion; emphasize monitoring and interventions with primary technical reports.

## 3. Regimes and displacement (C12-C20)

- [x] C12: Shorten the Section III heading to Constructive and Destructive Regimes.
- [x] C13: Elaborate the practical motivation for diagnostics available during training.
- [x] C14: Give the section opening a connected argument instead of listing variables.
- [x] C15: Remove the redundant step-length phrase.
- [x] C16: Remove unexplained "first-order" language from the intuitive gradient paragraph.
- [x] C17: Explain gradient norm as local slope magnitude and the positive individual-step contribution.
- [x] C18: Emphasize destructive interference at its first appearance in the decomposition.
- [x] C19: Explicitly factor inner products into magnitudes and directional alignment.
- [x] C20: State how constructive/destructive behavior follows from the update equations and shapes net parameter change.

## 4. Curvature, interpretation and a separate diagnostic (C21-C27)

- [x] C21: State the curvature subsection's goal before deriving it.
- [x] C22: Remove premature interpretation while defining curvature/noise terms.
- [x] C23: Box the consecutive-cosine/curvature relation.
- [x] C24: Explain the full-batch sign relation and the local regime boundary at eta*kappa=1.
- [x] C25: Separate qualitative parameter-overwriting/forgetting intuition from the optimization interpretation; remove alpha notation and check the literature connection.
- [x] C26: Create a subsection on estimating directional curvature without evaluating the Hessian; explain online cost and the large-batch connection.
- [x] C27: Give the estimator its precise GD/SGD interpretation and transition to the noise subsection.

## 5. Sampling and cumulative retention (C28-C31)

- [x] C28: Compact the noise argument; retain the nonpositive term, batch dependence and fixed-state attenuation, removing the general two-state cosine formula and "noise energy" wording.
- [x] C29: Motivate R by asking how much traveled parameter distance remains as net change.
- [x] C30: Remove the second displayed R-squared equation and explain cumulative interactions in prose.
- [x] C31: Add the orthogonal-step numerical example, distinguishing geometric retention from measured learning benefit.

## 6. Loss-increase conditions (C32-C37)

- [x] C32: Rename the subsection around loss decrease and spike formation.
- [x] C33: Use capital Delta for consecutive loss differences consistently in the active derivation and appendix.
- [x] C34: Open with the question of when destructive training produces an upward loss excursion.
- [x] C35: Develop the joint interpretation of cosine and gradient-norm growth.
- [x] C36: Box the condition, give the equivalent inner product and explain both necessary conditions with compact numerical examples.
- [x] C37: Keep the curvature form as a concise connection; distinguish the regime boundary from the loss-increase boundary and an individual increase from a recovered spike.

## 7. Controlled experiments and Figure 1 (C38-C41)

- [x] C38: State the experimental questions and matched setup in consistent regime terminology.
- [x] C39: Rename Section IV around constructive/destructive regimes.
- [x] C40a: Overlay retained eta*lambda_max measurements in Figure 1, preserving loss/diagnostic arrays, LR colors, smoothing and scientific provenance.
- [x] C40b: Explain the correspondence between oscillations, destructive alignment, norm growth and falling R.
- [x] C40c: Explain the alternating backward-lag profiles, including their nonmonotone envelope where present.
- [x] C40d: Explain the observed mini-batch attenuation without assuming a universal monotonic response.
- [x] C40e: Remove the endpoint table from compilation; keep its source and the rate/setup information needed by readers.
- [x] C41: Apply the argument across all of Section IV and remove avoidable page gaps.

## 8. Verification and handoff

- [x] Independently verify changed mathematical implications and Figure 1 values/indexing.
- [x] Record one disposition for every C01-C41, including how suggestions were interpreted.
- [x] Check that protected sections/assets and the feedback PDF are unchanged.
- [x] Compile, inspect every rendered page and resolve unintended layout/reference problems without restoring removed appendix floats.
- [x] Update manuscript/evidence records and commit only this task's changes locally.

## Scientific interpretation rules for this pass

The author's scientific direction governs the revision. Make the positive
argument explicit while distinguishing what each observation measures:
negative cosine describes parameter-direction interference; forgetting requires
retention of learned behavior to be assessed; low R measures geometric
retention of path length; the GD curvature diagnostic is directional, while
the requested spectral comparison uses the largest eigenvalue. These
distinctions belong beside the relevant interpretation, rather than as repeated
negative qualifications throughout the paper. No new training is needed.
