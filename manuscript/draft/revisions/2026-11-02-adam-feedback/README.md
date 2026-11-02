# Adam-section revision, 2026-11-02

Status: complete. The 17-page review PDF passes numerical, build and visual
checks. All six comments and the clarified central claim are addressed.

The author supplied six annotations on pages 9--10 of the preceding review.
Their unchanged PDF is preserved as `source-comments.pdf`; `comments.json`
records all comments and original annotation metadata. Editable work stays in
`manuscript/draft/`; build/render scratch stays in ignored `tmp/pdfs/`.

The revised section centers the author's clarified claim:

> In Adam, raw-gradient dynamics provide information about impending loss
> excursions beyond recent loss history, even though Adam transforms those
> gradients before updating the parameters.

This states the scientific idea generally. The experimental paragraph and
appendix separately specify the evidence: nine controlled full-batch arms,
one initialization per case, selected constant rates, and an exploratory
forward-time comparison. This scope is not embedded as a restriction in the
central claim. Demonstrating an improved learning-rate intervention remains
distinct from the incremental-information result.

## Dispositions

| Comment | Action |
| --- | --- |
| A01 | Remove AdamW and clipping discussion from the Adam section; retain only the Adam moment/update definitions needed for interpretation. |
| A02 | Lead with raw-gradient information beyond recent loss, show a forward-test comparison, and connect projected change and gradient/momentum growth to developing loss excursions. State the algebraic dependence of the threshold/sign/norm observations and preserve the distinction between temporal evidence and causal feedback. |
| A03 | Delete the old “Raw Gradients as Observations of the Traversed Loss” subsection and its generic Taylor exposition. |
| A04 | Delete the old “Momentum Growth, Rotation and Adaptive Scaling” subsection; give momentum a short, concrete role alongside the new measurements. |
| A05 | Replace Pythia/MiniPile illustrations and both old main-text tables with existing Analysis036 evidence, two compact vector figures and an incremental-prediction table. Original analyses and table sources remain preserved. |
| A06 | Use a compact empirical presentation: fixed design, cross-case observations, quantitative incremental information, raw spike examples, interpretation. Put detailed clocks, forecast protocol, hardware and limitations in the appendix. |

The latest steering explicitly leaves the SGD batch-size figure unchanged.
Its PDF, source arrays and section text are untouched by this revision.
Abstract, introduction and conclusion now reflect the Adam claim and current
experiments instead of the superseded language-model/schedule illustrations.

## Evidence and reproduction

[Analysis036 Observation009](../../../../analyses/036-2026-11-02-controlled-adam-diagnostics/observations/009-manuscript-raw-gradient-warnings.md)
records sources, captions, reduction, coverage, uncertainty and interpretation.
The two new figures retain 66 series / 53,777 values from the existing
verified histories. The forecast table contains 12 rounded scalars. All
original evidence, previous figures and the earlier 28-comment revision
remain preserved; there is no new training, cloud use or cost.

From the repository root:

```text
python manuscript/draft/revisions/2026-11-02-adam-feedback/01_prepare_figures.py
python manuscript/draft/revisions/2026-11-02-adam-feedback/02_build_and_verify.py
```

The generator creates the two compact Analysis036 figures, copies them to the
active draft and writes the forecast table. The verifier rebuilds all arrays
from raw histories, checks state clocks and source hashes, independently
rechecks temporal labels/scores, retains the preceding GD/SGD/math checks,
and compiles/renders the manuscript. Numerical/build results are in
`verification.json`; final visual acceptance and deliverable hash are in
`review.json`. The review deliverable is `revised-manuscript.pdf`; the usual
ignored compiled copy is also at `manuscript/draft/bare_jrnl.pdf`.
