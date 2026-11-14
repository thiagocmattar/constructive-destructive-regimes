# Closure tasks for the current paper plan

- **Authority:** [PLAN.md](../PLAN.md). This is an implementation index, not a second scientific plan.
- **Current phase:** the agreed H051 matched matrix and paper assets are complete. Manuscript rewriting/integration is the next phase; original draft and supplied inputs remain preserved.

| Plan step | Status | Work / acceptance |
|---|---|---|
| 1. Integrate evidence | Pending manuscript work | Revise I–III in place; retain IV's four figures; integrate V-A/B and the four V-C families. Figures 5–9, two algorithms, outcome/cost tables and captions are prepared. |
| 2. Matched intervention comparison | Complete | 36 initialization-0 conditions, MLP 512–800/ResNet 256–576/Transformer 544–928, every requested setting, qualified shared forks and exact native replay. Historical comparisons remain separate. |
| 3. Method reporting | Complete for this scope | Full loss/stage/update/activity histories, fixed past calibration, five full held-out states, 108 gradient/update audits, 36 terminal-gradient replays and 48 terminal held-out split replays. Timing and memory limits are explicit. |
| 4. Conditional extensions | Deferred | Initializations 1/2, cap x LR, forecasting, quiet-window/mode specificity, causal-source identification, stronger method comparisons and stochastic/scale transfer. Required only for the corresponding stronger claim. |
| 5. Final manuscript checks | Pending manuscript work | Abstract/conclusion/citations, compile/render, reproducibility package and TNNLS route/format. No contact or submission is authorized. |

- **Current package:** [selected V-C figures](../evidence/05-adam/c-intervention/figures/README.md), [matched observation](../evidence/05-adam/c-intervention/observations/65-016-matched-intervention-matrix.md), [captions](../evidence/05-adam/c-intervention/captions.tex), [compute closeout](../evidence/05-adam/c-intervention/data/matched-compute_closeout.json).
- **Future experiments:** use the [ablation TODO and separate ETC/costs](../evidence/05-adam/c-intervention/TODO_ABLATIONS.md). Continue Analysis 065; changed scientific inputs get new run IDs.
- **Maintenance:** run `python manuscript/consolidation/work/package_evidence.py --verify`. Rebuild only after reviewing intended source/catalogue changes. Historical numbering/prose is not automatically manuscript content.
