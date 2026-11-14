# Evidence package in paper order

- **Current authority:** [PLAN.md](../PLAN.md). The six top-level evidence notes are now aligned with its four contributions and closure sequence.
- **Contents:** 120 packaged assets from 119 unique source files, plus four regenerated compact tables, four equation sets, two algorithms and eight bibliography additions.
- **Provenance:** research sources remain canonical. Copies preserve bytes except recorded transformations: observation headers/links, one report dependency path, and the standalone q50 heatmap page extraction.
- **Editable material:** the consolidated equation/algorithm/table snippets have been syntax-checked. Original draft TeX is a source snapshot with its original input paths; it is not a standalone manuscript build.
- **14 November figures:** seven revised views are prepared and checked. Figure 6 now shows the larger ResNet excursion at 415 over steps 390–450; its [current verification](05-adam/b-tracing/data/resnet_peak_figure_verification.json) supplements the [initial seven-view checks](figure_revision_verification.json). The SVD companion retains its original ResNet 281–311 fit. [Current captions and timing](FIGURE_NOTES.md), [editable captions](figure_captions.tex). Original manuscript and intervention PDFs are preserved.
- **Readiness:** “prepared main/support” identifies the refreshed views; “main base” or “assembly component” still needs final composition/caption work. Historical comparators retain their original matching limitations.

| Folder | Use |
|---|---|
| [01-introduction](01-introduction/README.md) | Current introduction and contribution map |
| [02-related-work](02-related-work/README.md) | Current bibliography, eight additions, exact predecessor roles |
| [03-regimes](03-regimes/README.md) | Interaction/curvature/noise/loss equations and assumptions |
| [04-gd-sgd](04-gd-sgd/README.md) | Figures 1–4 and HVP, finite-step, all-lag/batch evidence |
| [05-adam/a-characterization](05-adam/a-characterization/README.md) | Compact Figure 5, sign/severity tables, complete forward audit |
| [05-adam/b-tracing](05-adam/b-tracing/README.md) | Assembled Figure 6, separate temporal companion, mode controls, Algorithm 1 and original report |
| [05-adam/c-intervention](05-adam/c-intervention/README.md) | Figures 7–10: LR → beta2 → clipping → distributed q50, outcome table and Algorithm 2 |
| [06-conclusion](06-conclusion/README.md) | Current conclusion and closing claims |
| [appendices](appendices/README.md) | Original derivations/diagnostics and supplied decision letter |

- **Audit notes:** [claims](CLAIMS.md), [argument map](ARGUMENT_MAP.md), [literature](LITERATURE.md), [author inputs](AUTHOR_INPUTS.md), [freshness review](REPOSITORY_REVIEW.md).
- **Asset selection:** [ARTIFACTS.md](ARTIFACTS.md). Full source/hash records: [copy_manifest.json](copy_manifest.json).
- **Checks:** [source/link/PDF checks](verification.json), [editable TeX checks](snippet_verification.json), [selected figure review](visual_review.json); rebuild/verify using the [packaging script](../work/package_evidence.py).
- Runs289–291 supply the new conventional model coverage; their canonical raw evidence remains in the owning runs/Analysis065. This package assembles that evidence with the previous studies. A matched four-family benchmark remains pending. Original report figure numbers remain historical; the PLAN assigns final paper slots.
