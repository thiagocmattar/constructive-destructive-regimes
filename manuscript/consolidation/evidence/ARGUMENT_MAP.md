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
- **Figures 7–9:** [four-family loss matrix](05-adam/c-intervention/figures/17-matched-intervention-loss.pdf) → [spike/learning tradeoff](05-adam/c-intervention/figures/18-matched-spike-learning-tradeoff.pdf) → [matched q50 heatmaps](05-adam/c-intervention/figures/19-matched-q50-stage-heatmaps.pdf). Rows retain LR → beta2 → clipping → cap. q25/q75 are sensitivity settings of the same cap method; q50 stays the preselected main intervention.
- **Algorithms 1–2:** tracing and local cap supplied as Markdown and TeX, using the common state clock.
- **Tables:** next-step signs, ≥10% severity, current spectral controls, [all 36 matched outcomes](05-adam/c-intervention/tables/matched-outcomes.csv) and [component costs](05-adam/c-intervention/tables/matched-costs.tex). Earlier intervention tables remain explicitly historical.
- **Equations:** III, V opening/A, V-B and V-C each have an equation file; original TeX snapshots remain alongside.
- **Closure:** use [TASKS.md](../work/TASKS.md), which mirrors the plan rather than opening a separate scientific agenda.
