# Observation004: publication view of the GD result

Figure: [cifar10_gd_diagnostics_paper.pdf](../figures/cifar10_gd_diagnostics_paper.pdf).
This compact 6.8-by-5.65-inch serif version supports the requested
[GD formulation report](../../../manuscript/2026-10-11-restatement/README.md).
It changes presentation only. The current six-panel GD figure is preserved
byte-for-byte; Adam figures and frozen run inputs are unchanged.

Caption: Full-batch GD on the same CIFAR-10 5k tanh MLP initialization, with
learning rates 0.004, 2/300, 0.01 and 0.02. (a) Training loss; (b) consecutive
raw-gradient cosine; (c) eta times the directional curvature estimate;
(d) eta times the largest algebraic Hessian eigenvalue, with a dashed reference
at two; (e) full-prefix raw B/A; (f) raw-gradient norm on log Y.
All curves are unsmoothed. Hessians are measured every 100 updates; the other
diagnostics are measured every update. The three larger rates enter a nearly
alternating-gradient regime, with the scaled directional curvature estimate near two and B/A falling sharply,
while training loss decreases over the horizon.

Sources: frozen Runs059-062 production metrics, curvature records, summaries
and final evaluations, reduced by [04_reduce_and_verify.py](../04_reduce_and_verify.py).
The [source verification](../verification.json) and
[results](../results.json) remain unchanged. The generator verifies retained
NPZ hashes and matches all 24 plotted X/Y arrays exactly to the
[current GD manifest](../gd_figure_manifest.json). Raw scalar histories and
derived NPZ files remain outside Git; no new gradient histories are reconstructed.

Reduction: eta*kappa_hat_t = 1 - rho_t*||g_t||/||g_(t-1)||, with zero gradient
noise under full-batch GD; B_t/A_t = ||sum_(s=1)^t g_s||/sum_(s=1)^t ||g_s||.
No rolling mean, separate-factor averaging, zero padding or temporal thinning.
The undefined first cosine and curvature estimate remain undefined. Maximum curvature uses
the largest algebraic eigenvalue directly multiplied by eta. Loss and Hessian
indices are parameter states; gradient indices name updates.

Coverage: all 20,000 updates in each run, all 656,810 trainable coordinates,
19,999 defined adjacent pairs per run, loss states 0..20,000, and 201 Hessian
samples per run at states 0,100,...,20,000. The last gradient belongs to state
19,999; final loss and curvature belong to state 20,000. Colors identify the
same learning rate in every panel. Dense per-update curves are rasterized at
300 dpi inside the PDF; text, axes and sparse curves remain vector elements.
Rasterization does not alter the input arrays or remove samples.

Uncertainty and caveats: one frozen initialization and finite horizon, without
confidence bands. Eigenpairs pass relative residual <= 0.005 and have only
100-step temporal resolution. The directional curvature estimate is a finite-step
quantity, not generally the pointwise maximum Hessian eigenvalue. The reference
at two is local to quadratic GD. A low B/A describes path-length cancellation,
not small absolute displacement or necessarily slow loss reduction. These
diagnostics do not establish a spatial spiral, forgetting or generalization.

Generator: [11_plot_gd_paper.py](../11_plot_gd_paper.py).
Panel (c) is titled **Directional curvature**, retaining the estimated quantity
$\eta\widehat\kappa_t$ and the same plotted arrays.
The [publication manifest](../gd_paper_figure_manifest.json) records source and
plot-array identities, figure dimensions and PDF hash. The report's
[build/visual manifest](../../../manuscript/2026-10-11-restatement/gd-formulation-manifest.json)
records the embedded figure and every report-page render from its latest build, checked for
legibility, clipping, equation layout, captions and references.
The title revision was independently rendered with Poppler and visually checked;
all 24 plotted X/Y arrays match the previous publication figure exactly. Its
render hash and review are recorded in the publication manifest. The embedded
manuscript figure is refreshed on the next report rebuild.
