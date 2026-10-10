# Observation002: logarithmic raw norms and LR overlays of cumulative geometry

Question: expose raw-gradient norm trajectories across all seven LRs with a logarithmic Y axis, and compare each cumulative metric across LRs on dedicated panels. This user-requested display revision introduces no new training, data reduction or empirical claim.

Sources: the unchanged `../comparison_data.json`, SHA256 a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739, containing all712 updates for Runs021-027 and their original source hashes. Coverage/reductions and one-seed limitations are in [Observation001](001-seven-lr-results.md). Generating script: `../../10_plot_lr_geometry.py`, using the unchanged Analysis008 `08_plot_simplified.py` values/smoothing helpers. [Palette, source/generator hashes and figure metadata](../requested_figures_data.json); [PDF and render verification](../verification.json).

## Revised six-metric view

[Combined metrics](../figures/metrics_by_learning_rate.pdf): the existing3x2 view, now with logarithmic Y only in panel(e), raw-gradient individual vector norms. Every displayed raw norm is positive and finite, so no observation is discarded for log plotting. Panel(f), momentum norms, remains linear. Loss is a trailing25-update token-weighted mean; adjacent cosines use trailing25-update medians with light raw traces. Norm curves are unsmoothed. Display updates1-711; partial update712 remains in the source/reductions. Undefined initial adjacent cosines remain gaps.

All seven LR colors now use reversed plasma (`plasma_r`), sampled uniformly in log10(peakLR), using colormap positions0.10-0.95. Lowest LR is yellow, highest LR dark purple. The seven sampled CIELAB lightness values decrease strictly with LR, verified in the generating script. Dashed metric curves identify original Spark runs; solid curves identify the new RTX4090 runs.

## New cumulative view

[Geometry across LRs](../figures/geometry_across_learning_rates.pdf):2x2, row-major panels(a) raw-gradientg A and B;(b) stored momentum m A and B;(c) raw-gradient B/A;(d) momentum B/A. Each panel overlays all seven LRs using the same sequential palette as the metric view. The top row uses solid A=sum of norms and dashed B=norm of sum; both lines for one LR share its color. Bottom-row B/A lines are solid. All four Y axes are linear and each cumulative-norm panel scales independently. Raw B/A retains0-1.05; momentum B/A now automatically fits the complete displayed curve range with5% vertical padding. All curves are unsmoothed; no median-based shading or temporal threshold is introduced.

Display revision, 2026-10-09: the user requested removal of panel(d)'s fixed0-1.05 range. Its limits use the minimum and maximum across all seven momentum B/A curves, including early updates, with5% padding on each side. No curve is cropped. The source SHA and all other PDFs are unchanged; panel(c)'s limits are unchanged. Actual limits, complete-curve coverage checks, revised generator/PDF hashes and visual QA are recorded in the linked metadata and verification files. The tighter vertical range emphasizes differences within momentum B/A; vertical distances cannot be compared directly against panel(c).

Caption/legend: color identifies LR in increasing order from light yellow to dark purple. S=Spark, R=RTX4090 in the global LR legend; A/B style legends appear in both top panels. g is raw before clipping; m is actual uncorrected Adam exp_avg updated from clipped gradients, before preconditioning. A and B are vector-stream geometry, not actual parameter displacement.

Uncertainty/caveats: one seed1234, identical frozen initialization and token order, full finite-budget passes and complete token-weighted validation. Hardware/runtime differ across original/new groups; LR also scales AdamW decay. Curves do not establish convergence, stability, generalization or a causal momentum benefit. Scientific inputs and numerical results remain unchanged. The existing within-LR geometry PDF retains its per-column linear top-row scales.
