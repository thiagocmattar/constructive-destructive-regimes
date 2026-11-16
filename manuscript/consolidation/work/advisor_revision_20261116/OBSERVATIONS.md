# Advisor revision applied

16 November 2026. The author approved the editorial revision and explicitly rejected cross-correlation in item 18. The revised manuscript and supplement are ready for the author's next reading pass. No training experiment or remote execution was performed.

## Changes to review

- Main PDF: 20 pages. Supplement: 31 pages. Both compile without warnings, unresolved references, or overfull/underfull boxes.
- Abstract, introduction, related work, and repeated explanations were shortened. A consistent lexical count of main-section TeX decreases from 10,724 to 10,171 tokens (about 5.2%); this includes TeX command names and is not a prose-only word count. Page count remains unchanged because the retained figures, tables, and appendices occupy substantial space.
- Section V uses `Q_t` for Adam's raw-gradient projected response. It remains distinct from curvature along the actual parameter increment. The overview and diagnostic table use the new symbol; Supplement S1 explains the historical label on retained supporting plates.
- Section VI uses `s_{t,j}^ell` for the cap score, retaining `c_t^ell` for representation-change cosine. The supplement uses the same score letter and gamma for the scaling factor.
- The eight-update calibration prefix and frozen budgets are explicit. The current inputs, scores, and factors vary; the budget does not.
- Figure 7 overlays complete-objective loss on a secondary axis in each signed-coefficient panel, marking the window maximum. No cross-correlation was computed or added.
- Figure 8 retains all 36 points, grays the conventional settings, and highlights unchanged Adam and the three cap quantiles. Full values remain in Table III.
- Supplement S3 reports issued warnings, correct warnings, and recall for all seven fixed rules in all three models, taken from the retained diagnostic CSV. No classifier was trained.
- Section VII is now **Discussion and Conclusion**, connecting functional measurements to intervention outcomes while separating exploration, compactness, sparsity, and biological analogies from established results.

## Disposition of all meeting blocks

Numbers refer to the stable [action list](../../draft/ADVISOR_ACTION_ITEMS_2026-11-16.md).

| Item | Resolution |
|---|---|
| 1 | Retained functional motivation; weight-space programming remains meeting context. |
| 2 | Classical secant provenance explicit; contribution and literature comparisons narrowed. |
| 3 | Central Flows architectural question retained in the shorter introduction and related work. |
| 4 | Removed repeated prose; preserved definitions, equations, and supporting checks. |
| 5 | Four contributions retained and connected to the intervention result. |
| 6 | Figure 1 smoothing/coverage and Table I retained; scientific claims checked. |
| 7 | Introduced `Q_t` and explicitly distinguished applied-step curvature. |
| 8 | Preserved highest-rate selection, initialization normalization, and separate event thresholds. |
| 9 | Preserved decision timing, imbalance baselines, strict quadratic scope, and nonlinear counterevidence. |
| 10 | Explicit fixed rules with no trained classifier; added full warning/recall coverage. |
| 11 | Retained fixed inputs, flattened changes, dimensions, and aggregate-cosine meaning. |
| 12 | Clarified parameter displacement versus norm change; retained orthogonal example and useful-learning caveat. |
| 13 | Compressed SVD explanation; retained uncentered squared movement and reconstruction meaning. |
| 14 | Retained measured timing and independent normalization limits; no propagation mechanism asserted. |
| 15 | Added exact output ranks 5/20/15 and corresponding 51/61/49-change windows to the narrative. |
| 16 | Explicitly rejected causal-fraction interpretation of temporal overlap. |
| 17 | Explicit signed coefficient versus nonnegative singular-value/eigenvalue distinction. |
| 18 | Added only secondary loss axes and peak markers; cross-correlation dismissed. |
| 19 | Explicitly separated per-unit cap from SVD mode suppression and neuron concentration. |
| 20 | Renamed the score consistently in the equations and supplement. |
| 21 | Explicit frozen prefix calibration; no rolling controller introduced. |
| 22 | Defined upward variation as the sum of positive changes, distinct from maximum loss and peak count. |
| 23 | Retained all-cap mean/upward-variation improvements, ResNet q50 endpoint exception, and cost limits. |
| 24 | Simplified Figure 8 without selecting away comparison settings. |
| 25 | Retained Figure 9 and its common RMS normalization and q25 selection disclosure. |
| 26 | Multiobjective framing remains an open question; no dead-neuron percentage or necessity claim. |
| 27 | Sparsity/activity/exploration studies remain future work; no new experiment required for this handoff. |
| 28 | Biological analogy qualified; no bias/threshold mechanism or homeostasis claim added. |
| 29 | Central finding leads the abstract and conclusion; four contributions remain. |
| 30 | Developed implications and limitations in the closing discussion. |
| 31 | Prepared local review PDFs; delivery to Cristiano and later Braga remains with the author. |

## Figure observations and provenance

The [figure manifest](figure-manifest.json) records source hashes, plotted-array checks, loss clocks and peaks, and output hashes. The [generator](02_figures.py) reproduces the three revised figures from retained local evidence.

| Figure | Sources and reduction | Observation, coverage, and limits |
|---|---|---|
| 5 | Existing overview builder and frozen `figure_revision_20261114/plotted_arrays.npz`; label-only scientific change. | All 24 plotted arrays unchanged. Highest-rate trajectory per model; selected cases and one initialization. No uncertainty estimate added. |
| 7 | Existing MLP/Transformer temporal data and Analysis 063's ResNet 390-450 results/motion files; unchanged SVD coefficients, spectra, overlaps, and RMS scaling. Added full-objective loss without smoothing on the same completed-step clock. | All 36 prior arrays unchanged; six loss x/y arrays added. Peak steps are 474/415/562 for MLP/ResNet/Transformer. Windows contain 51/61/49 changes. Left/right axes are independently scaled; this is descriptive timing within a retrospective SVD fit, not causal or prospective evidence. Caption identifies curves, axes, and peak markers. |
| 8 | `matched-results.json` and `matched-curves.npz` from the frozen intervention evidence. Final loss is the endpoint; upward variation sums positive consecutive differences. | All 36 points unchanged and checked against saved losses and the prior presentation manifest. No point selection, new ranking, or new uncertainty estimate. Log x and symlog y preserve zero variation. Caption retains unequal strengths and one-initialization limits. |

The [warning report](warning-coverage.json) records the original CSV hash and output hash. All 21 rows satisfy their confusion-matrix totals and recall identities. Adding Table S3 advances subsequent supplement table numbers automatically; cross-references resolve.

## Verification

[Verification record](verification.json) checks all 44 main-paper equations including appendices, allowing only the declared symbol changes; existing section labels; unchanged numerical content of the two edited main tables; 50 figure-source hashes; unchanged protected draft files; and both final PDF hashes. Existing scientific evidence, supplied transcript, feedback originals, bibliography, Section IV, and appendix sources remain unchanged.

All 20 main pages and 31 supplement pages were inspected in rendered contact sheets. Revised figure previews and the new coverage table were inspected separately at larger scale. Figure 9 was reduced to 97% text width and Figure 9/Table III were allowed on a proper float page to avoid an overfull page after reflow. No scientific figure content changed there. Final renders show readable axes, legends, captions, and tables without overlap or clipping.

The [primary-source audit](SOURCE_AUDIT.md) records the literature checks. No required scientific choice remains unresolved for this editorial handoff. Future experiments remain future work.

## Reproduction

Run from the repository root:

```powershell
python manuscript/consolidation/work/advisor_revision_20261116/02_figures.py
python manuscript/consolidation/work/advisor_revision_20261116/03_warning_coverage.py
python manuscript/consolidation/work/advisor_revision_20261116/04_build.py manuscript
python manuscript/consolidation/work/advisor_revision_20261116/04_build.py supplementary
python manuscript/consolidation/work/advisor_revision_20261116/05_verify.py
```

`01_revise.py` is the historical, one-time prose transformation from the ignored pre-edit snapshot; do not rerun it on the finished revision. Its exact replacements are retained in `text-replacements.json`. Abstract, introduction, conclusion, supplemental reporting, and float adjustments were edited directly and are recorded in Git. Verification against the pre-edit draft requires the local `scratch/baseline/` snapshot whose hashes are retained in `baseline-manifest.json`. Build logs, page renders, and contact sheets are ignored scratch artifacts.
