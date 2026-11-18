# Current argument and artifact map

- **Authority:** [PLAN.md](../PLAN.md), four contributions, Sections I–VI; updated 14 November 2026.
- **Supersedes:** the earlier C1–C8/M01–M24 layout and obsolete figure numbering. Its useful scientific content is retained in the section packages.
- **Thesis:** connect gradient interaction, curvature/noise, finite-step loss and functional movement; test selective control without equating reversal with failure.

| Contribution | Specific question | Existing answer / evidence | Paper home |
|---|---|---|---|
| 1. Local reversal and cumulative progress | How can adjacent opposition coexist with net learning progress? | Exact all-pair displacement and R; nine GD cases have positive total pair interaction, seven negative adjacent interaction. | [III](03-regimes/README.md), [IV Figures 2–3](04-gd-sgd/README.md) |
| 2. Directional curvature from existing gradients | What scalar is recovered, how accurate is its pointwise interpretation, and how does sampling enter? | Exact smooth-GD path average, error bound, BB credit and noise term; 1,188 MLP directional-HVP pairs. | [III equations](03-regimes/equations.md), [IV supporting validation](04-gd-sgd/README.md) |
| 3. Reversal, amplification and loss consequences | Which joint condition explains an incoming rise, and what information precedes future Adam rises? | Quadratic-GD condition with nonlinear residual; exhaustive Adam timing/severity audit. | [III](03-regimes/README.md), [V-A](05-adam/a-characterization/README.md) |
| 4. Functional tracing and selective control | Where does motion appear, and which intervention preserves useful progress? | Tracing algorithm, stage-specific findings, spectral controls, exact MLP loss accounting and the complete 36-condition, four-family matched intervention comparison. | [V-B](05-adam/b-tracing/README.md), [V-C](05-adam/c-intervention/README.md) |

## Display decisions

- **Figures 1–4:** retain current PDFs, in the current order.
- **Figure 5:** high-LR overview exists; attach the compact population summary. Complete sensitivity curves remain supporting evidence.
- **Figure 6:** the grouped loss/cosine/RMS display and separate temporal-structure companion are prepared and verified; ResNet uses the larger 390–450 event in both.
- **V-C main figures, final numbers pending:** [loss-variation/learning tradeoff](05-adam/c-intervention/figures/24-main-intervention-tradeoff.pdf) → [matched q50 heatmaps](05-adam/c-intervention/figures/25-main-q50-functional-motion.pdf), with the [compact table](05-adam/c-intervention/tables/matched-main-outcomes.tex) covering all 36 conditions. The [three-page draft](../draft/archive/2026-11-18/intervention-review.pdf) passes numerical and final-size rendering checks. The [full loss matrix](05-adam/c-intervention/figures/17-matched-intervention-loss.pdf) supports it in the supplement. Rows retain LR → beta2 → clipping → cap; q50 stays preselected, with q25/q75 as sensitivity settings.
- **Algorithms 1–2:** tracing and local cap supplied as Markdown and TeX, using the common state clock.
- **Tables:** next-step signs, ≥10% severity, current spectral controls, and [all 36 matched outcomes](05-adam/c-intervention/tables/matched-outcomes.csv). The completed main intervention table has 12 setting rows and three model groups with PV, mean/final training and test loss. Detailed outcomes and [component costs](05-adam/c-intervention/tables/matched-costs.tex) are supplementary, with endpoint/test/cost limitations explicit in the draft. Earlier intervention tables remain historical.
- **Equations:** III, V opening/A, V-B and V-C each have an equation file; original TeX snapshots remain alongside.
- **Closure:** use [TASKS.md](../work/TASKS.md), which mirrors the plan rather than opening a separate scientific agenda.
