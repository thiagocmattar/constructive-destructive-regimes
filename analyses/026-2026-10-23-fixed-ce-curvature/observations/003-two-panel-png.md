# Observation 003: two-panel FP32 PNG

User-requested presentation revision, October23,2026. Only panels(a) training
loss and(b) eta*lambda_max from [Observation002](002-focused-fp32-figure.md)
are exported to [PNG](../figures/fixed_ce_curvature_fp32_loss_lambda.png),
3420x1500 pixels at300dpi. No new scientific measurement or training.

Sources: the unchanged [verified FP32 figure data](../fp32_figure_data.json),
bound by its [original manifest](../fp32_figure_manifest.json). The generating
[script](../14_plot_fp32_two_panel_png.py) checks all44 seed/median curves for
exact agreement with those arrays. The new [manifest](../fp32_two_panel_png_manifest.json)
binds the script, data and PNG and records numeric/image/visual checks.

Coverage, reductions and uncertainty are unchanged: native/fixed FP32 training,
ten paired seeds each,10,000 updates, all480 training pairs. Loss uses100-state
arithmetic means within each seed, with actual final-state dots. Eta=0.001
multiplies all23 saved largest algebraic Hessian eigenvalues per run. Thin lines
are seeds; thick lines are pointwise seed medians. Both Y axes are logarithmic.
The common FP64 diagnostic is retained, including the true native seed7 final
loss spike. No seed or measured curvature state is removed. All curve values
and endpoint dots are copied unchanged from the four-panel version.

Caption: Fixed cross-entropy under FP32 AdamW training. (a) Accurate training
loss; (b) eta*lambda_max. Modular addition mod31, width64 Transformer,480 training
pairs, ten paired initializations and10,000 updates per arm. Orange/dashed:
native CE; blue/solid: fixed CE. Thin curves: seeds; thick curves: seed medians.
Loss uses100-state means plus terminal dots; curvature uses23 measured states.
Both arms share the same FP64 loss/Hessian diagnostic.

Interpretation and limits are unchanged: the large attained-curvature difference
accompanies lower attained loss; matched-loss controls and seed uncertainty
remain in the [report](../REPORT.md). These are ordinary Hessian measurements,
not an AdamW stability bound. Connecting sparse spectra does not measure
intervening spikes. This PNG contains only the two requested panels; the earlier
PDFs and full control evidence are retained.

Reproduce with `python analyses/026-2026-10-23-fixed-ce-curvature/14_plot_fp32_two_panel_png.py`.
After visually checking the actual PNG, append `--accept-reviewed` to record review.
