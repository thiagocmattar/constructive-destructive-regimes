# Observation008: applied surgical revision of Section IV

Date: 2026-10-30. The user authorized application of the
[eight-edit proposal](../../../manuscript/2026-10-30-section-iv-surgical-proposal/README.md).
This implements the proposal using the completed nine-condition evidence and
Observation007's retained scalar audit. No new training or measurements are
introduced.

## Manuscript changes

- IV now opens with the distinction between consecutive reversal, cumulative
  geometry and objective progress, positioned relative to the cited EOS work.
- IV-A retains the matched comparison and measurement definitions. Detailed
  preparation, precision, accumulation and spectral protocols move to Appendix D.
- IV-B calls longer-timescale coexistence an illustration and points to the
  separate local loss-approximation audit.
- IV-C reports positive final all-pair interaction in all nine trajectories
  despite negative cumulative lag-one interaction in seven, and increasing
  absolute displacement with decreasing R within each model case.
- IV-D reports the retrospective loss comparison, including the Transformer
  LR0.2 sign disagreements and threshold sensitivity. The approximation remains
  distinct from the exact displacement identities and prospective forecasting.
- Table I replaces its spectral column with gradient-sum displacement. Its
  first six columns and all nine conditions remain.
- Appendix D retains its raw-cosine text/figure and adds explicit temporal
  definitions, three generated evidence tables and spectral context. The
  PyTorch word-language-model reference is pinned to the recorded revision
  `13acec6d7c78dacd5e1fe9b0b4a325e1d39abc15`; the bibliography URL is rendered
  using the standard url package.
- The two approved Section III prose edits qualify learned-information and
  path-length-retention interpretations. Its equations are unchanged.

The original three main figures, raw-cosine PDF and all scientific arrays are
unchanged. A local samepage block keeps the sensitivity result from splitting
across the full figure page. The broader legacy Section V revision remains
outside this applied package.

## Sources and reductions

`results.json` supplies the original endpoint and spectral values.
`section_iv_review_checks.json` supplies gradient-sum displacement, path length,
temporal interactions, sign counts and normalized median residuals. The latter
remains the historical review record, including its pre-revision manuscript
hashes; it was not regenerated to overwrite that snapshot.

`08_build_manuscript_assets.py` now generates four nine-row tables directly from
these records. It checks the audit script and retained metrics hashes before
generating assets, and records the frozen reference-source provenance in
`manuscript_assets.json`. Display rounding is four decimal places for ordinary
endpoints and four significant digits for the appendix interactions/residuals.

The temporal columns include the factor of two and use raw-gradient units;
they are not scaled by eta squared. The final total is the squared norm of
the gradient sum minus the sum of squared gradient norms. The retrospective
audit uses t=2..3000, 2999 pairs per condition, with the final update excluded
because no subsequent gradient was recorded. Every threshold column reports
opposite signs / eligible count, and the residual median uses all pairs.
All previous single-initialization, objective-coverage, temporal-dependence,
finite-step and floating-point interpretation limits are retained.

## Verification and visual review

The extended `09_verify_manuscript.py` independently parses all 36 table rows
and reconstructs the added quantities from the hash-checked raw metrics.
For the loss check it uses the equivalent gradient-dot expression, separately
from the audit script's q-based expression. All 26991 comparisons, threshold
denominators and displayed residuals agree. It checks displacement, temporal
accounting, coverage labels, all nine spectral endpoints, and preservation of
the original 54 Y arrays, Section III equations and Appendix D raw-alignment
block. `manuscript_verification.json` records the current output and source
hashes.

The 18-page working draft compiles with no overfull boxes, unresolved
references/citations or duplicate labels. Nonfatal template/legacy underfull
and float-placement warnings remain in the ignored build log. Section IV
occupies pages 6-9; the three main figures are on pages 7-8 before Section V.
Appendix D occupies pages 15-17, with the raw figure on page 16 and all three
new tables on page 17. The added pinned-code reference appears on page 18.

All 18 pages were rendered with Poppler. Pages 4-9 and 15-18 were inspected
individually for text, mathematical notation, tables and figure legibility;
the remaining pages were checked in a contact sheet for reflow defects.
The page-7/page-9 transition now falls between complete paragraphs around
the full-page figures. No labels overlap curves or table entries.

Current compiled output is `manuscript/artifacts/draft/bare_jrnl.pdf`.
Pre-edit snapshots, build logs and renders are retained under ignored
`manuscript/artifacts/draft/section-iv-application/`. Raw evidence, executed
inputs, prior observations and supplied manuscript originals are preserved.
