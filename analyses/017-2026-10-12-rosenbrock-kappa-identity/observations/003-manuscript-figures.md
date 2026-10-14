# Observation003: Rosenbrock publication figures

Figures: [full trajectories and objective contours](../figures/rosenbrock_trajectories_paper.pdf)
and [eight-panel diagnostics](../figures/rosenbrock_diagnostics_paper.pdf).
They support the results section after the Adam formulation in the
[report](../../../manuscript/2026-10-11-restatement/README.md).

Sources: immutable Runs055-057, through the source receipts loaded by
[kappa.py](../kappa.py). All source-history, config and verification hashes are
checked. The original Analysis017 reductions and PDFs remain unchanged. The
[new results](../paper_results.json) record scalar reductions and derived array
hashes; new vector/projection arrays are retained in ignored `artifacts/` files.
There is no new optimizer run, seed or cloud execution.

Setup: f(x,y)=(1-x)^2+100(y-x^2)^2; AdamW, 2,000 float64 updates from (0,1),
LR 0.01/0.0031622776601683794/0.001, betas (0.9,0.95), epsilon 1e-8,
decoupled decay 0.1, exact analytic gradients, no clipping or schedule.
Momentum means the stored uncorrected first moment m. Adaptive updates use mhat.

Trajectory caption: Full optimization paths on objective level curves, retaining
all 2,001 states. The square marks the common start, the star the optimum (1,1),
and circles final states. Arrows follow recorded update order. Shared LR colors
identify the same trajectory in both figures. The domain contains every state;
no separate zoom, projection or spatial interpolation of the trajectory is used.

Diagnostics caption: (a) objective, log Y; (b) raw-gradient and momentum norms,
log Y; (c) raw cosine; (d) momentum cosine; (e) eta*largest objective-Hessian
eigenvalue; (f) raw-gradient projection estimate; (g) momentum projection
estimate; (h) full-prefix B/A for both streams, raw dashed and momentum solid.
Panels c,d,f,g use complete 100-pair trailing means; other series are unsmoothed.
The ratio limits follow the observed range (approximately 0.810..1.012).

Reduction: the raw estimate is eta*kappa_hat=1-rho*||g_t||/||g_(t-1)||,
identical to the original eta overlay before and after smoothing. For momentum,
rearrange Eq.25 with zero gradient noise:

    eta*kappa_hat_m = a_t + (1-c_t*||m_t||/||m_(t-1)||)/(1-beta1)
                   = m_(t-1)^T (g_(t-1)-g_t) / ||m_(t-1)||^2.

The tracking term a_t is retained; merely replacing g by m in the raw formula
would give a different statistic. The cosine is invariant to bias correction,
but the accumulation ratio and momentum projection depend on the moment scale.
Both vector sums and norms use stored m, consistently with the report.

Checks: all 5,997 pairs and 22,800 complete plotted means are verified. A
quadratic example checks the momentum inversion independently. On every actual
pair, the estimate equals the eta-scaled mixed Hessian contribution plus its
known AdamW decay contribution minus the projected nonlinear remainder.
Loss, gradients, Hessians/eigenvalues, moment recurrence, update convention,
cosines, norms and B/A are reconstructed from retained coordinates/vectors.
All 30 diagnostic series and three full paths have actual plotted-array hashes.

Observed final and late-100 values:

| LR | Final objective | Raw B/A | Momentum B/A | Raw cosine | Momentum cosine | eta*kappa_hat_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.0009722631 | 0.8223 | 0.9209 | -0.6539 | 0.6427 | 5.6179 |
| 0.00316227766 | 0.0001179581 | 0.9520 | 0.9612 | -0.6522 | 0.6629 | 5.1679 |
| 0.001 | 0.0000110471 | 0.9665 | 0.9671 | -0.6688 | 0.6585 | 5.1181 |

Interpretation: raw-gradient reversal occurs earlier at larger LR, while late
momentum alignment remains positive on average. Full-prefix accumulation stays
high; late adjacent reversal does not imply severe cancellation of the complete
trajectory. Smaller LR attains lower objective at this fixed horizon.

Coverage and uncertainty: three deterministic 2D trajectories, no ensemble
uncertainty. Loss and Hessian use state t=0..2000; gradients/moments use updates
1..2000 and their gradients are evaluated at theta[t-1]. Adjacent pairs use
updates 2..2000, with the first full mean at 101. Pair decomposition uses the
Hessian at theta[t-2]. The first pair remains undefined; all runs have positive
norm denominators. There is no thinning or averaging of separate norm/cosine
factors. Projection estimates along AdamW include adaptive-step, decay and
finite-step effects; neither is a general largest-eigenvalue estimator or an
AdamW stability test. The B/A ratio is not actual adaptive-update displacement.

Generator: [05_plot_paper.py](../05_plot_paper.py). The
[manifest](../paper_figures_manifest.json) records source, array and PDF hashes,
scales, smoothing, preserved original PDFs and Poppler render/visual review.
