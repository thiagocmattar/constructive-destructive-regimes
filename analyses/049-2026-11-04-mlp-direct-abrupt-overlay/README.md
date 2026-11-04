# Analysis049: direct decay and abrupt restoration overlay

Completed 2026-11-04. User-requested direct overlay of two existing trajectories. This explicit request replaces baseline-only pairing for this figure. No training, new run, seed or hypothesis. [Vector figure](figures/mlp_direct075_vs_half_abrupt075.pdf), [source values](plotted_values.csv), [verification](verification.json). Generate with01_plot.py.

Sources: shared Run223 prefix through511; Run226 cosine32_three_quarters from512..800; Run224 cosine32_half_return075 from512..800. Exact paths/hashes are recorded in verification.json. Same MLP/CIFAR-10, seed0, full5000-image objective, physical1000 partitions, Adam and Spark. Existing run-level exact warm-state/replay qualifications are preserved. These are complete schedule trajectories, not matched later parameter/optimizer states.

## Observation and caption

- Blue: cosine32 reduces base LR0.001 to0.00075 over512..544, then holds it.
- Magenta: cosine32 reduces LR to0.0005 over512..544, holds until650, then abruptly increases to0.00075.
- Both paths start from the same authenticated warm512 and have the same applied LR after650, but their intervening model and Adam-state histories differ. No time shift or alignment to spike peaks was applied.
- Five rows directly overlay log training loss, raw rho, raw gradient norm, normalized projected gradient change and applied/base LR. Curves are solid, unsmoothed native observations. There is no unchanged-baseline curve in this requested overlay.
- Display448..800 includes common pre-fork context (gray shading). Vertical lines mark512 and650. All data are on Spark; there is no hardware transition. Endpoint800 is unapplied; the LR curve closes the final799..800 interval.

Caption: MLP/CIFAR-10 seed0, direct comparison of cosine32-to0.75x (blue) and cosine32-to0.5x followed by abrupt restoration to0.75x at650 (magenta). The paired schedules share their initial state and later LR, but retain different training histories. Raw curves are unshifted and unsmoothed. Read loss and diagnostics jointly; this single selected window does not establish a universal stability condition or identify the mechanism of oscillations. Final excursions may be censored at800.

Validation:100 artifact size/SHA checks,3528 plotted value/source checks and complete state indexing. One-page vector PDF has no raster objects; rendered PNG was visually inspected for curves, axes, legend and clipping. No scientific input, manuscript, cloud resource or training run changed.
