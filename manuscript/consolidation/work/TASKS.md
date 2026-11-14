# Closure tasks for the current paper plan

- **Authority:** [PLAN.md](../PLAN.md). This is an implementation index, not a second scientific plan.
- **Current phase:** the complete [17-page manuscript](../draft/manuscript.pdf) and [33-page supplement](../draft/supplementary.pdf) are ready for author review. The agreed H051 matrix is incorporated; the active draft and supplied inputs remain preserved.

| Plan step | Status | Work / acceptance |
|---|---|---|
| 1. Integrate evidence | Complete inside consolidation | Author-guided I–VI revision close to the active draft, consistent t notation, nine main figures, three tables, two algorithms, reconciled references, and separately numbered supplement. Original title, theory spine, and IV's four figures retained. The prior three-page V-C excerpt is preserved. |
| 2. Matched intervention comparison | Complete | 36 initialization-0 conditions, MLP 512–800/ResNet 256–576/Transformer 544–928, every requested setting, qualified shared forks and exact native replay. Historical comparisons remain separate. |
| 3. Method reporting | Complete for this scope | Full loss/stage/update/activity histories, fixed past calibration, five full held-out states, 108 gradient/update audits, 36 terminal-gradient replays and 48 terminal held-out split replays. Timing and memory limits are explicit. |
| 4. Conditional extensions | Deferred | Initializations 1/2, cap x LR, forecasting, quiet-window/mode specificity, causal-source identification, stronger method comparisons and stochastic/scale transfer. Required only for the corresponding stronger claim. |
| 5. Final manuscript checks | Technical checks complete; author/venue decisions remain | Abstract, conclusion, citations, all 144 intervention cells, mathematical identities, source preservation, warning-free compilation, and all-page visual review pass. Author review and the existing TNNLS route/current-format issue remain before submission. No contact or submission is authorized. |

- **Current package:** [selected V-C figures](../evidence/05-adam/c-intervention/figures/README.md), [matched observation](../evidence/05-adam/c-intervention/observations/65-016-matched-intervention-matrix.md), [captions](../evidence/05-adam/c-intervention/captions.tex), [compute closeout](../evidence/05-adam/c-intervention/data/matched-compute_closeout.json).
- **Future experiments:** use the [ablation TODO and separate ETC/costs](../evidence/05-adam/c-intervention/TODO_ABLATIONS.md). Continue Analysis 065; changed scientific inputs get new run IDs.
- **Current revision audit:** [sources, adversarial dispositions, and reproduction](readability_revision_20261114/OBSERVATIONS.md); [numerical checks](readability_revision_20261114/verification.json); [visual-review record](readability_revision_20261114/visual_review.json). These are provenance, not additional required author reading.
- **Maintenance:** run `python manuscript/consolidation/work/package_evidence.py --verify`. Rebuild only after reviewing intended source/catalogue changes. Historical numbering/prose is not automatically manuscript content.
