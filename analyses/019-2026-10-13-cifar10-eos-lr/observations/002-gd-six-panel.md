# Observation002: six-panel GD diagnostics

The user requested a simpler GD-only view and archival of the four detailed
figures. This is a presentation revision of the completed experiment, using
Runs059-062 without new training or changed scientific inputs. The dictated
"unsmoothed trailing loss (ling Y scale)" is interpreted as unsmoothed training
loss on linear Y. The initial figure used complete 100-pair trailing means
for cosine and eta-kappa. At the user's subsequent request, the current figure
removes those means and shows both pair metrics unsmoothed.

Figure: [cifar10_gd_diagnostics.pdf](../figures/cifar10_gd_diagnostics.pdf).

Caption: Four matched full-batch GD learning rates (0.004, 2/300, 0.01, 0.02)
on the same CIFAR-10 5k tanh MLP initialization. (a) Unsmoothed training loss,
linear Y; (b) raw-gradient adjacent cosine and (c) learning-rate-scaled
gradient-identity kappa, each unsmoothed; (d) learning rate times
the largest algebraic objective-Hessian eigenvalue, with the local GD reference
2; (e) full-prefix raw-gradient cumulative displacement B/A; (f) unsmoothed
raw-gradient norm, log Y. Colors identify the same learning rate in every panel.

Sources: frozen Runs059-062 production `metrics.jsonl`, `curvature.jsonl`,
`summary.json` and final evaluations, reduced by
[04_reduce_and_verify.py](../04_reduce_and_verify.py). The
[per-run verification](../verification.json) records source hashes and the
size/SHA256 of each retained `derived/RunNNN.npz`. The new generator checks those
derived-file identities before plotting. Full scalar histories and NPZ arrays
remain outside Git; no dense gradient history is claimed.

Reduction: define g_t at theta[t-1], rho_t = cosine(g_(t-1), g_t), and
eta*kappa_hat_t = 1 - rho_t*||g_t||/||g_(t-1)||, with the explicit noise term
zero, using the same per-pair identity as
[Analysis017](../../017-2026-10-12-rosenbrock-kappa-identity/README.md).
No rolling mean, zero padding, averaging of separate factors, or temporal
thinning is used. Each pair metric shows all 19,999 defined values per run,
starting at update 2. The undefined first pair remains undefined.
B/A = ||sum_(s=1)^t g_s|| / sum_(s=1)^t ||g_s||
uses the entire prefix. Panel (d) multiplies lambda_max directly by eta, with
no division by 2 and no preconditioning. Only the largest eigenvalue is shown.

Coverage: all 20,000 updates per run and all 656,810 trainable coordinates.
Loss includes pre-update states 0..19,999 plus the independently evaluated
final state 20,000. Gradient panels use update indices 1..20,000; the last raw
gradient therefore belongs to state 19,999. Hessian spectra cover states
0..20,000 every 100 updates (201 samples per run). These state indices are
preserved rather than shifted to create an apparent pointwise equivalence.

Observation: at the three larger learning rates, the adjacent cosine becomes
strongly negative, eta-kappa approaches 2, raw norms increase and B/A falls.
The transition occurs earlier at larger LR in this matched cohort. LR 0.004
retains a cosine near +1, eta-kappa near zero and B/A near 0.76 at the horizon,
while eta*lambda_max rises to about 1.68. Thus the gradient-derived quantity
and the largest Hessian eigenvalue need not coincide. Exact numerical results
and event indices remain in the [report](../REPORT.md).

Uncertainty and caveats: one frozen seed, finite horizon, no confidence bands.
Curvature estimates use the complete 5k objective and pass iterative relative
residual <= 0.005; they have only 100-step temporal resolution. The gradient
quantity measures a directional secant along the trajectory, whereas the
Hessian value is the largest algebraic eigenvalue. The unsmoothed curves expose
per-step variations that the original 100-pair trailing means suppressed.
The dashed 2 is a local GD quadratic reference, not a global nonlinear
stability guarantee. Loss, alignment and B/A alone do not certify convergence
or generalization. See Observation001 for the unchanged data and optimizer setup.

Generator and checks: [09_plot_gd.py](../09_plot_gd.py) records all 24 actual
plotted arrays, checks kappa and B/A arithmetic, source hashes, and all 159,992
displayed finite pair values (two metrics, four runs) in
[gd_figure_manifest.json](../gd_figure_manifest.json).
[07_pdf_qa.py](../07_pdf_qa.py) with `--simple-gd` checks and renders the PDF;
[gd_visual_verification.json](../gd_visual_verification.json) records the
reviewed page hash, render and text bounds.

Archive: the four original PDFs are retained in `figures/archive/` with their
original sizes and SHA256 values. The
[archive manifest](../archive_manifest.json) maps their former paths to their
current locations. The original numeric and visual manifests are preserved.
