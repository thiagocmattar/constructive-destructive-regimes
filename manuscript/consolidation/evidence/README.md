# Evidence package in paper order

- **Current authority:** [PLAN.md](../PLAN.md). The six top-level evidence notes are now aligned with its four contributions and closure sequence.
- **Contents:** 151 packaged assets from 150 unique source files, plus regenerated compact tables, consolidated equations, two algorithms and bibliography additions. The new matched matrix adds seven PDFs (20 pages), editable outcome/cost tables and complete numerical provenance.
- **Provenance:** research sources remain canonical. Copies preserve bytes except recorded transformations: observation headers/links, one report dependency path, and the standalone q50 heatmap page extraction.
- **Editable material:** the consolidated equation/algorithm/table snippets have been syntax-checked. Original draft TeX is a source snapshot with its original input paths; it is not a standalone manuscript build.
- **Earlier IV and V-A/B revision, 14 November:** seven revised views are prepared and checked. Both selected ResNet tracing figures now use steps 390–450 around the peak at 415. The [fresh temporal fit and verification](05-adam/b-tracing/data/resnet_temporal_figure_verification.json) match the [loss/motion view](05-adam/b-tracing/data/resnet_peak_figure_verification.json); original fits remain historical. [Current notes](FIGURE_NOTES.md), [editable captions](figure_captions.tex). MLP/Transformer tracing panels and original manuscript are preserved; the intervention refresh is described below.
- **Readiness:** “prepared main/support” identifies the refreshed views; “main base” or “assembly component” still needs final composition/caption work. Historical comparators retain their original matching limitations.

| Folder | Use |
|---|---|
| [01-introduction](01-introduction/README.md) | Current introduction and contribution map |
| [02-related-work](02-related-work/README.md) | Current bibliography, eight additions, exact predecessor roles |
| [03-regimes](03-regimes/README.md) | Interaction/curvature/noise/loss equations and assumptions |
| [04-gd-sgd](04-gd-sgd/README.md) | Figures 1–4 and HVP, finite-step, all-lag/batch evidence |
| [05-adam/a-characterization](05-adam/a-characterization/README.md) | Compact Figure 5, sign/severity tables, complete forward audit |
| [05-adam/b-tracing](05-adam/b-tracing/README.md) | Assembled Figure 6, separate temporal companion, mode controls, Algorithm 1 and original report |
| [05-adam/c-intervention](05-adam/c-intervention/README.md) | Figures 7–9: matched four-family loss, spike/learning tradeoff and q50 heatmaps; all-setting support, outcome/cost tables and Algorithm 2 |
| [06-conclusion](06-conclusion/README.md) | Current conclusion and closing claims |
| [appendices](appendices/README.md) | Original derivations/diagnostics and supplied decision letter |

- **Audit notes:** [claims](CLAIMS.md), [argument map](ARGUMENT_MAP.md), [literature](LITERATURE.md), [author inputs](AUTHOR_INPUTS.md), [freshness review](REPOSITORY_REVIEW.md).
- **Asset selection:** [ARTIFACTS.md](ARTIFACTS.md). Full source/hash records: [copy_manifest.json](copy_manifest.json).
- **Checks:** [source/link/PDF checks](verification.json), [editable TeX checks](snippet_verification.json), [earlier IV/V-A/B figure review](visual_review.json); rebuild/verify using the [packaging script](../work/package_evidence.py).
- **Consolidation V-C draft, 14 November:** [three IEEE-style pages](../draft/archive/2026-11-18/intervention-review.pdf), [editable fragment](../draft/05c-interventions.tex), two polished main figures (24/25), one compact 12-row table and Algorithm 2. All values and final-size rendering are verified; full-paper integration remains open. Original H051 plots and active manuscript files are preserved.
- **Matched intervention evidence is complete:** [H051 / Runs 293–295](05-adam/c-intervention/observations/65-016-matched-intervention-matrix.md), 36 conditions / 11,904 updates, full fixed-window measurements and terminal-state replay audits. [Captions](05-adam/c-intervention/captions.tex), [figure verification](05-adam/c-intervention/data/matched-visual_verification.json), [retention/cleanup](05-adam/c-intervention/data/matched-compute_closeout.json). Earlier intervention figures and observations remain historical; the PLAN assigns current paper slots.
