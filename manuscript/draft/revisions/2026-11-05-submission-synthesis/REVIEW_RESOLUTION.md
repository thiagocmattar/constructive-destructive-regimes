# Review resolution and readiness assessment

5 November 2026. This record distinguishes repaired manuscript defects from scientific limitations that writing cannot remove. Internal simulated reviews do not constitute journal decisions.

## Reviewed argument and evidence

The paper diagnoses temporal gradient interaction by measuring adjacent cancellation, cumulative motion and loss progress separately, with explicit optimizer-vector semantics. The original constructive/destructive object remains central; "destructive" denotes a component of geometric cancellation at the reported horizon.

The final title is **Constructive and Destructive Training Regimes: Diagnosing Gradient Interaction**. The revision uses the existing GD/SGD, Adam, Pythia and matched-intervention evidence. There is no new training, cloud spending, controller, sparsity method or orthogonalization proposal. The previous Adam experiment plan remains future work for broader claims.

The positive diagnostic payoff is specific:

- Similar near-antiparallel GD cosines hide different within-pair cancellation and residual coherence. The Transformer cancels the greatest fraction locally among the three highlighted high-rate paths but retains the largest fraction overall.
- All nine GD paths have positive total cross-step interaction despite seven negative adjacent totals. A small path ratio and a negative adjacent cosine cannot establish negative total interaction.
- The Adam rise intervals have exclusively descending first-order applied steps. Measuring the actual increment excludes an explanation based solely on locally uphill directions.
- Clipping Adam's raw input can increase actual motion: the selected MLP mean and maximum step norms both rise. The selected ResNet response differs. Stream identity and observation horizon therefore affect what an intervention comparison establishes.

## Review sequence

1. **Independent synthesis and positioning.** [Evidence inventory](reviews/evidence-inventory.md), [theory/novelty assessment](reviews/theory-novelty.md), and [argument advisor](reviews/argument-advisor.md). These assessments identified missing close literature, an overly broad stability promise, a disproportionate spike-prediction emphasis and existing evidence that could close the argument.
2. **First adversarial review of the rewrite.** [Methods](reviews/round1-methods.md) and [framing](reviews/round1-framing.md). The reports were allowed to recommend major revision and identify enduring significance risks.
3. **Second review of the rebuilt paper.** [Methods](reviews/round2-methods.md), [framing](reviews/round2-framing.md), and [evidence/layout](reviews/round2-evidence-visual.md). Both scientific reviewers found no remaining actionable blocker within the chosen scope. Their reports preserve the reviewed PDF hash; a subsequent presentation-only rebuild is documented below.

## Concrete issue disposition

| Concern | Repair and verification |
| --- | --- |
| Title promises general SGD stability | Title, abstract, introduction and conclusion now name diagnostic interpretation. No parameter-norm or alignment-sign stability theorem is claimed. |
| Classical secant presented as a new estimator | Main text derives the exact BB1 reciprocal relation, with its nonzero-denominator scope. BB, gradient convergence diagnostics, hypergradients and trajectory-directionality literature are cited. |
| Prior norm-control objections unanswered | Related work now distinguishes Li/Liu and Lei's probabilistic guarantees from realized-path accounting. The discarded negative "batch distance" and lag-one truncation are not revived. |
| Local reversal conflated with total interaction | All-lag identity, zero-cross-term reference and paired factorization are adjacent in the theory. GD results explicitly compare negative adjacent totals with positive all-pair totals. |
| Little positive diagnostic payoff | GD text explains what C/D adds beyond saturated cosine. Intervention text uses verified actual-step norms to explain why clipping input is not equivalent to controlling motion. |
| GD quadratic relation silently generalized to Adam or forecasting | Loss subsection is retrospective, exact on quadratics and qualified by nonlinear error. Adam has a separate actual-step interpretation and smooth finite-step integral. |
| Full-GD Hessian identities applied automatically to ReLU | Measurement appendix names the ReLU models and limits classical-Hessian path identities across nonsmooth activation boundaries. |
| Adam mechanism/HVP claim uncalibrated | Detailed bridge moved to Appendix C, with assumptions and lack of neural calibration explicit. It does not headline the empirical contribution or require a new campaign. |
| Spike regressions dominate the paper | Four figures removed from compilation; all assets preserved. The negative held-rate-transfer result remains visible without a predictive claim. |
| Pythia raw accumulation read as actual displacement | Explicit distinction before the replicated raw-stream results. AdamW clipping, moments, rates and decay remain part of the update contract. |
| Probe exclusion stated too broadly | Restricted to the union of the three 350-update training prefixes in the coordinate-scaling study, with same-corpus scope retained. |
| Pythia schedule underspecified | Warmup, peak/floor and original 712-update clock are stated; the exact pre-step/clamped formula and code link are in the revision README and immutable run records. |
| Intervention outcomes pooled or horizon omitted | Table II names each study's objective, seed coverage, warm state and horizon. Mean, maximum, endpoint and late-window results remain distinct. |
| Intervention display values only compared with a hand-copied manifest | Verifier now independently reduces 16 retained scalar histories (4,732 rows), checks 140 numerical comparisons and 23 displayed claims, with 29 source hashes. |
| Figure state labels differ from text notation | Captions explicitly map the Adam figures' horizontal state count t to state s in the text; original figure bytes remain unchanged. |
| Uneven final reference columns / missing PDF metadata | Final reference break balanced at reference 16; title, authors, subject and keywords embedded. Final renders rechecked. |

## Numerical, source and layout verification

- Analysis056: 43 authenticated sources, 65 synthetic vector cases, 26,991 scalar pair checks, 546 SGD continuations and 2,304 backward pairs pass.
- Analysis057: 39 source hashes, nine independent event segmentations, 812 candidate boundaries, 127 severity events, 2,170 phase-update checks, 9,216 cross-alignment states and 112 independent QR fits pass. Its existing 2,096 mathematical cases also pass.
- Added formulation checks: 100 nontrivial polynomial cases verify inverse-BB1 and the actual-step loss integral, including indefinite Hessians and directions not proportional to the gradient. Maximum absolute discrepancies are approximately 2.22e-14 and 4.64e-15, respectively.
- The intervention verifier independently checks the raw reductions and all table statements; its output is saved in [evidence_verification.json](evidence_verification.json).
- The final PDF has 18 pages, seven figures and four tables, with no unresolved references/citations, multiply defined labels or overfull boxes.
- All 34 protected original figures, tables and supplied previous-submission files retain their baseline hashes.
- Every final page was visually inspected across the primary agent and evidence reviewer. The final PDF hash and page-image hashes are recorded in [visual_review.json](visual_review.json); the active PDF and revision review copy are identical.

The final rebuild after the two scientific reviews changed only caption notation, reference-column balance, embedded metadata and a trailing source blank line. It introduced no new result, numerical value, mathematical relation or scientific claim. The evidence reviewer rechecked the affected pages; the final whitespace-cleanup rebuild retained byte-identical renders on all 18 pages. The final artifact hash is recorded in the review addendum and visual manifest.

## Remaining scientific and submission judgments

The revision is ready for author assessment as a coherent diagnostic paper. Its remaining risks are substantive and should remain visible:

1. **Significance and novelty.** Norm expansions, secants, gradient products and productive oscillations are established. The contribution is the combined temporal accounting, quantitative characterization and optimizer-aware diagnostic consequences. A journal may still judge that advance incremental.
2. **Replication.** The main full-batch GD and Adam sweeps are conditional on one initialization per model. Three-block SGD/Pythia evidence cannot be transferred to Adam event replication; dependent events are not independent runs.
3. **Utility.** The paper establishes what measurements mean and how selected interventions change observed outcomes. It does not establish that a diagnostic policy improves future training decisions, an overhead advantage, or a universally beneficial intervention.
4. **Scope.** Parameter-space geometry depends on coordinates and is not a measure of functional learning, memory retention or a generalization guarantee.
5. **Venue packaging.** The current deliverable is a complete 18-page IEEE journal draft including appendices and references. The target venue, current submission-format/page rules, any supplemental separation and final author approval remain to be settled for the actual submission. No unverified hard limit or acceptance probability is asserted.

The stopping rule is to repair identifiable support, consistency and clarity defects, then expose the remaining scientific judgment to the authors. Repeating simulated reviews solely until a favorable acceptance vote appears would add no evidence. This revision reaches that stopping point without reopening the experiment tree.
