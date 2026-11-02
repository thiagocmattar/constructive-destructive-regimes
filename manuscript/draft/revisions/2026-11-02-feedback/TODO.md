# Annotated manuscript revision

Source: `source-comments.pdf`, preserved byte-for-byte from the supplied
`manuscript/artifacts/draft/bare_jrnl+coments.pdf` on 2026-11-02.
The 28 annotations, original page locations and highlighted passages are in
`comments.json`. C01--C28 follow PDF page/annotation order; C27 and C28 overlap.

## Working task

Produce a surgical revision with dense, interpretable prose that complements
the mathematics and evidence. Keep editable work, review inputs and the next
review PDF under `manuscript/draft`; keep temporary build/render files ignored.
The original PDF annotations reuse completed experiments. The subsequent
three-case SGD figure request requires additional measurements; see the open
extension below. No new training or cloud execution has started.

- [x] C01: Replace Section III's title with a title centered on the regimes.
- [x] C02--C03: Delete the two marked displacement sentences/clauses.
- [x] C04--C06: Interpret curvature as the change in the gradient's projection;
  explain the directional quadratic minimum and the threshold at one; introduce
  the measurable diagnostic after the interpretation. Preserve manuscript indexing.
- [x] C07--C08: Expand the preceding noisy gradient inline to derive the expected
  noise interaction; remove the marked normalized-expectation aside.
- [x] C09--C10: Condense the noise explanation. State precisely when independent
  noise attenuates either sign of alignment and why evolving SGD differs.
- [x] C11: Explain R near zero/one and expected low retention under sustained
  reversal or incoherence; distinguish relative retention from absolute motion.
- [x] C12--C15: Split the loss-change equation into magnitude, interference and
  noise terms; derive the joint norm/alignment condition and curvature threshold
  at two. Negative alignment makes the signed interference contribution positive.
- [x] C16--C19, C22: Give the empirical section a direct goal and one compact
  design paragraph. Use prose for interpretation; move implementation details
  to the existing appendix and retain the matched-comparison limits.
- [x] C20: Regenerate a 2-by-3 GD loss/projected-change figure. Explain the
  first-order connection to EoS without equating directional finite-step change
  with the largest Hessian eigenvalue or claiming a validated onset predictor.
- [x] C21: Regenerate a 3-by-3 cosine/norm/retention figure and explain reversal,
  growing steps and inefficient path retention alongside continuing progress.
- [x] C23: Assess Analysis035 and add a compact backward-lag comparison with
  explicit anchor, coverage and recomputation provenance.
- [x] C24: Add dataset to the endpoint table, use T notation, define late means,
  and use decimal formatting with a footnote for nonzero values rounded to zero.
- [x] C25: Remove the detailed loss-surrogate audit from the main narrative;
  preserve its equation and evidence in the existing appendix.
- [x] C26--C28: Reduce SGD to a subsection of the GD empirical section; make a
  2-by-2 figure for initialization and the 2,000-update state, with one sampled
  gradient curve per panel and explicit full-batch endpoint. Explain observed
  attenuation and the late-state retention exception. Briefly connect to
  implicit regularization without asserting an unmeasured generalization benefit.
- [x] Verify equations, numerical reductions, source hashes, labels, citations,
  all regenerated artwork and the complete compiled manuscript.
- [x] Record comment dispositions, evidence and remaining interpretation limits;
  commit only this task's completed changes.

## Scientific editorial decisions

The comments motivate interpretation, not new empirical laws. A small R does
not imply small absolute displacement, negative total all-lag interaction, or
slower objective progress. The existing largest-rate endpoints improve loss.
Independent noise at fixed states can attenuate either sign under concentrated
norms; the fixed-single-state formula alone only describes nonnegative expected
alignment. Actual SGD changes the trajectory as well. A noise-dominated mean
cosine near zero does not establish isotropic noise or exact orthogonality of
every pair. The loss-rise threshold is a local quadratic approximation, not a
guarantee for every nonlinear training step.

Completed: all 28 annotations addressed with dispositions in README.md.
The 16-page manuscript and four figures pass numerical and visual review.

## Additional feedback: three-case SGD figure

The user confirmed the existing three model/dataset cases, rather than three
distinct datasets. This supersedes the 2-by-2 layout specified above.

- [x] Check existing evidence: Analysis027 supplies the MLP continuations;
  Analysis020 supplies only frozen-state controls for the other two cases.
- [x] Specify the [2-by-3 figure and measurement extension](SGD_EXTENSION.md).
- [ ] Resolve the execution resource/budget and qualify the required inputs.
- [ ] Measure ResNet-32 and Transformer continuations at initialization and
  after 2,000 full-GD updates, with the declared replication and controls.
- [ ] Replace the compact figure with three case columns, two metric rows,
  and two starting-state curves in every panel; update the empirical prose
  from the observed results, without assuming attenuation in every case.
- [ ] Verify reductions, uncertainty, complete-objective losses, figure layout
  and the rebuilt manuscript; commit the completed extension.

The current review PDF remains a verified intermediate revision. Its SGD
figure has not yet incorporated this additional feedback.
