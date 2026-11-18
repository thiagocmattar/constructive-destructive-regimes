# Main intervention displays and V-C draft

- **Scope:** presentation of the completed H051 comparison, 14 November 2026. No new training, scientific inputs, windows, settings, smoothing or numerical reductions. All work stays in consolidation; the active manuscript remains preserved.
- **Review:** [integrated IEEE-style PDF](../../draft/archive/2026-11-18/intervention-review.pdf), [editable V-C fragment](../../draft/05c-interventions.tex). The [single plan](../../PLAN.md) remains the executive map. Figure/table numbers in the excerpt are local; Algorithm 2 retains the planned method numbering.
- **Sources:** the packaged [full-precision summary](../../evidence/05-adam/c-intervention/data/matched-results.json), [state/stage histories](../../evidence/05-adam/c-intervention/data/matched-curves.npz), [event records](../../evidence/05-adam/c-intervention/data/matched-events.csv), [held-out evaluations](../../evidence/05-adam/c-intervention/data/matched-heldout.csv) and [scientific observation](../../evidence/05-adam/c-intervention/observations/65-016-matched-intervention-matrix.md).
- **Generator:** [01_prepare.py](01_prepare.py). [Presentation manifest](data/presentation_manifest.json) records every plotted point, heatmap array hash, loss-series hash, fixed normalization, peak reference and table cell. Inputs and outputs carry SHA-256 identities. [Independent verification](verification.json), [visual review](visual_review.json) and [build/render script](02_build_review.py) accompany the final review.

| Main asset | Reduction and coverage | Message / boundary |
|---|---|---|
| [Tradeoff](figures/24-main-intervention-tradeoff.pdf) | All 33 treated points plus three Adam origins; exact full-window mean/PV changes; original adverse points retained. Nine-point ordinary labels; family shapes/colors; callouts move text only. | All nine cap/model combinations lower mean loss and positive variation in these cases. PV measures accumulated upward loss change, not event count. One initialization/model and unequal effective strengths. |
| [Functional motion](figures/25-main-q50-functional-motion.pdf) | Three full windows: MLP 512--800, ResNet 256--576, Transformer 544--928. Five/six/six stage endpoints; loss at every state; incoming cosine/RMS at every treated endpoint. Both arms use the same per-stage Adam RMS maximum. Color limits include all values. | Smaller extreme output jumps coexist with persistent reversals. ResNet cumulative squared movement increases; q50's weak whole-window PV effect and worse endpoint remain explicit. Stage labels are shortened, not aggregated. |
| [Compact outcome table](tables/matched-main-outcomes.tex) | Twelve setting rows and three model groups; 144 percentage cells and all 12 absolute Adam reference values. PV reduction is favorable when positive; loss changes when negative. | Mean/final/test outcomes are simultaneous. Bold labels the preselected q50 setting, not a winner. No confidence intervals inferred from correlated updates or settings. |

- **Captions:** the authoritative composed captions live beside the figures in the [V-C fragment](../../draft/05c-interventions.tex); the table carries its caption and notes. The heatmap caption defines incoming-step placement, shared denominators, stage abbreviations, output spaces and descriptive baseline-peak markers. Treatment starts at the common fork, never at the marker.
- **Draft argument:** local direct-displacement cap and its scope; matched four-family design; mean/PV tradeoff; persistent reversal and displacement magnitude; endpoint, held-out, cost and replication limits. All 36 conditions appear in the main table. Historical windows do not count as matched seed replications.
- **Supporting assets:** original matched PDFs 17--23, their exact tables and analysis sources remain unchanged. They retain full control trajectories, all-setting maps, five-state held-out coverage, activity and costs. The new main displays supersede 18/19 for manuscript placement, not as raw evidence.
- **Literature check:** the existing bibliography keys for [Adam](https://arxiv.org/abs/1412.6980) and [gradient norm clipping](https://arxiv.org/abs/1211.5063) were checked against the original sources on 14 November 2026. The retained-moment beta2 switch is this experiment's implementation; no from-initialization or novelty claim is attributed to those papers.
- **Remaining integration:** incorporate the reviewed fragment into the eventual complete draft and reconcile its figure/table/equation numbering, supplementary references and surrounding V-A/B narrative. This deliverable does not alter the current manuscript or claim broader empirical coverage.

## Reproduction

Run from the repository root, using the retained packaged `matched-curves.npz` and its accompanying JSON/CSV files:

```powershell
python -X utf8 manuscript/consolidation/work/intervention_manuscript_20261114/01_prepare.py
python -X utf8 manuscript/consolidation/work/package_evidence.py --build
python -X utf8 manuscript/consolidation/work/intervention_manuscript_20261114/02_build_review.py
python -X utf8 manuscript/consolidation/work/intervention_manuscript_20261114/03_verify.py
python -X utf8 manuscript/consolidation/work/package_evidence.py --build --verify
```

The figure generator needs NumPy/Matplotlib; the build needs IEEEtran, pdfLaTeX, BibTeX and PyMuPDF. LaTeX auxiliaries and rendered PNGs remain in this work directory's ignored `scratch/` folder. The three-page review includes two figures, the compact table, Algorithm 2, four equations and two references. Final-size rendering passed without LaTeX warnings, clipped labels or callout boxes covering plotted points. Numerical checks cover 144 recomputed summary metrics, all 144 compiled table cells, 12 absolute references, 33 treated scatter points, 22,656 heatmap values and 1,990 loss values.
