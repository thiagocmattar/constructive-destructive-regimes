# Conservative Section V readability revision

15 November 2026. Applies the approved review of the [Adam introduction](../../draft/sections/05-adam.tex), [diagnostic results](../../draft/sections/05a-characterization.tex), and [representation analysis](../../draft/sections/05b-tracing.tex) in the [main manuscript](../../draft/manuscript.pdf).

## Changes

- Immediately after the Adam projection identity, the text distinguishes the raw-gradient response statistic from curvature projected onto the actual applied update. The notation and both formulas remain unchanged.
- Observed Adam excursions and gradient behavior precede the cited mechanism. The text identifies that mechanism as compatible with the observations, rather than isolated by these measurements. Unmeasured basin-transition and momentum explanations are removed from the post-excursion paragraph.
- The single-step warning target (at least 0.1%) is explicitly distinguished from the plotted excursions exceeding 10%. The existing rule name, threshold, timing, performance results, and table are retained.
- The site definition now includes the existing output-coordinate convention: raw MLP outputs and all-class/all-word log probabilities for ResNet and Transformer. This information was already present in the appendix.
- Negative site cosine is explicitly an aggregate over flattened coordinates; it does not require every feature or reference input to reverse. The MLP description is limited to the observed negative aggregate signs at all measured sites.
- The RMS interpretation begins with the independent site normalization. The mathematical reason tanh cannot enlarge matched-coordinate changes is stated separately: its derivative has magnitude at most one. The normalized maps are not presented as quantifying cross-site attenuation. No new quantitative attenuation result is claimed.
- The ResNet/Transformer paragraph retains all measured depth and timing observations, while removing the untested backpropagated-error mechanism and stating the descriptive scope.

The [paragraph review](PARAGRAPH_REVIEW.md) contains nine sequential entries, each recording the read passage, diagnosis, proposed solution, and replacement before the edit. The source text grows by 45 whitespace-separated tokens including TeX across the three files.

## Preserved formulation and evidence

All 11 displayed equations (25--35), 19 labels, three figure blocks and captions, the tracing algorithm, and the diagnostic table are preserved. The SVD and temporal-overlap exposition is unchanged. Existing numerical tokens are retained; the only additions repeat the existing 0.1% and 10% event thresholds to clarify their difference. No experiments, thresholds, measured coordinates, numerical results, or measurement methods changed.

The intervention section, other scientific sections, bibliography, figures, tables, appendix, and supplement are unchanged. The changed draft files are the three Section V source files, the rebuilt PDF, and the draft README.

The [verification record](verification.json) replays the audit to reproduce the final source files, checks equations, figures, algorithm, labels, inputs, citations, SVD exposition, and numerical tokens, resolves cross-references, and verifies 54 protected draft files and five pre-existing modified research records.

## Build and visual review

The [build record](build.json) reports 22 pages with no LaTeX, undefined-reference, overfull, or underfull warnings. Pages 8--12 were inspected individually; all changed pages 8--14 were checked in contact sheets. No clipping or overlap was observed. Pages 1--7 and 15--22 render identically to the previously reviewed baseline. Downstream layout differences are reflow only.

From the repository root:

```powershell
python manuscript/consolidation/work/section5_readability_20261115/02_build.py
python manuscript/consolidation/work/section5_readability_20261115/03_verify.py
```

Verification uses the task's local ignored snapshots and manifests. The committed audit is independently readable; its baseline is commit `aef62bc`. Auxiliary files, edit inputs, and rendered pages remain in ignored scratch storage. A subsequent rebuild needs renewed visual inspection before refreshing the manual-review statement.
