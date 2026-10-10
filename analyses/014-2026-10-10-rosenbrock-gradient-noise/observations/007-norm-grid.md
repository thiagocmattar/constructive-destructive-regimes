# Observation007: matched raw-gradient and momentum norms

The user requested this presentation after numerical execution. Scientific
inputs and histories are unchanged. [PDF](../figures/gradient_momentum_norms_grid.pdf).

Sources: the [cohort](../cohort.json) maps all 15 sigma/LR combinations to
complete local histories, SHA-256 pinned by [verification](../verification.json)
and [figure manifest](../figures_manifest.json). Sigma0 reuses Runs038-040;
nonzero rows use Runs041-052. [Results](../results.json) retain the per-condition
norm reductions. Numerical verification independently computes all norms from
the recorded 2D vectors; all source/history/PDF hashes are rechecked at closeout.

Caption: **Raw-gradient and stored-momentum norms at matched learning rates and
gradient noise.** The 5x3 grid has columns LR1e-2, geometric midpoint
0.0031622776601683794, and LR1e-3; rows sigma0,0.01,0.1,1,10. Every cell plots
||g_t|| in solid blue and ||m_t|| in dashed orange at all updates1-2000.
Raw g_t=grad f(theta_previous)+xi_t is the estimator used by AdamW; stored
first moment m_t=0.9*m_previous+0.1*g_t precedes bias correction and adaptive
scaling. Shared logarithmic Y limits show the large early gradients and small
late values on comparable axes. Steps use linear X; no smoothing or thinning.
The sigma0 curves are deterministic controls. All nonzero conditions use one
matched normal-draw sequence scaled by sigma, seed1234. The exact analytic
gradient, noise, adaptive direction and actual displacement are separate streams.

Reduction: Euclidean norm over both coordinates, no added epsilon; all observed
raw/momentum norms are positive. Coverage is 30 curves, each with 2,000 values,
60,000 plotted norms in total. No seed averaging, uncertainty bands or inferred
minibatch-noise distribution. At sigma10, late mean raw norms are
13.283/12.766/12.588 in column order, versus momentum2.944/2.881/2.832.
At sigma0 the corresponding means are raw0.323/0.105/0.072 and
momentum0.038/0.013/0.057. These are descriptive means over updates1001-2000;
the PDF displays unsmoothed values. The norm difference does not itself establish
convergence, stability, objective decrease or a causal filtering benefit.

Generating script: [02_plot.py](../02_plot.py), `norm_grid`, using verified
histories; [03_record.py](../03_record.py) validates sources, prefixes and PDF
dimensions/text/bounds, and renders the page. [Figure verification](../figure_verification.json)
records rendered review. See [method](../METHOD.md) for the matched optimizer
recipe and [comparison observation](006-sigma-comparison.md) for loss and
adjacent-cosine outcomes. One paired realization supplies no ensemble inference.
