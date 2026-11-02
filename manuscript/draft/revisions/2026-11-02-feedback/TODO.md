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
extension below. Run214 and Run216 completed the new measurements on Runpod with qualified
Spark assistance.

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
- [x] Verify pinned inputs, prepare the runner and numerical checks, and measure
  local runtime/memory in Run214 / Analysis037 (mechanics only).
- [x] Qualify both complete short worker queues, full-batch replay, sampled
  forks, independent vector/index checks and copied-artifact integrity.
- [x] Start detached Runpod execution with qualified workers and an independent
  spending/deadline guard. The user extended the original four-hour limit to
  six hours while retaining US$40 total, including retries and retrieval.
  Hard deadline: 2026-11-02 16:34:56 São Paulo / 19:34:56 UTC.
- [x] Finish execution and retrieve/verify every worker before cleanup; see
  [Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/README.md).
- [x] Qualify the idle Spark GB10 with unchanged frozen inputs, complete-gradient
  agreement and short replay; retain explicit runtime/hardware provenance.
- [x] Complete the Spark 200-update runtime bridge, verify its 32 small-group
  handoffs and retrieve its complete raw inventory.
- [x] Measure ResNet-32 and Transformer continuations at initialization and
  after 2,000 full-GD updates, with the declared replication and controls.
- [x] Replace the compact figure with three case columns, two metric rows,
  and two starting-state curves in every panel; update the empirical prose
  from the observed results, without assuming attenuation in every case.
- [x] Verify reductions, uncertainty, complete-objective losses, figure layout
  and the rebuilt manuscript; commit the completed extension.

The review PDF now includes the verified three-case figure. All 348 new
continuations and both helpers are verified; all 13 task Pods/disks are deleted.
Provisional cost US$26.30; 4.663 hours within US$40/six hours.
Code, results and completed checks are recorded in
[Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/README.md).

## Additional feedback: figure layout and archive

- [x] Separate GD panel letters, column titles and learning-rate legends;
  remove excessive header spacing in both Analysis024 composites.
- [x] Use ordinary decimal training-loss ticks while preserving log scales.
- [x] Add a second backward-lag row showing the exact lags 1--32 and update
  its caption, provenance and independent numerical checks.
- [x] Archive the 26 unused figure copies, preserve bytes/hashes and update
  inactive TeX references. Leave the five compiled figures in `draft/figures`.
- [x] Rebuild, inspect figures and affected manuscript pages, and commit.
- [x] Group Table I dataset/model cells with `\multirow`; verify that its 45
  displayed values are unchanged.
