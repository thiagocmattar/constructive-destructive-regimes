# Observation003: Adam full-horizon and early-step diagnostics

The user requested an Adam counterpart of Observation002, adding momentum
adjacent similarity, momentum cumulative B/A, momentum norm, and the gradient
norm divided by momentum norm. They also requested a separate early-step zoom.
Both figures reuse completed Runs063-066; no training, scientific-input change
or new run is involved. The unspecified early interval is the first 2,000
updates (10% of the fixed horizon), stated before plotting.

Figures:

- [Full 20,000 updates](../figures/cifar10_adam_diagnostics.pdf)
- [First 2,000 updates](../figures/cifar10_adam_diagnostics_early.pdf)

Caption for both: Four matched full-batch Adam learning rates (5e-5, 1e-4,
2e-4, 4e-4) on the same CIFAR-10 5k tanh MLP initialization. (a) Unsmoothed
linear training loss; (b) raw-gradient adjacent cosine; (c) bias-corrected
momentum adjacent cosine; (d) eta-kappa from the raw gradients; (e) eta times
the largest algebraic eigenvalue of the ordinary objective Hessian; (f) raw
full-prefix B/A; (g) momentum full-prefix B/A; (h) raw-gradient norm; (i)
momentum norm; (j) raw-gradient norm divided by momentum norm. Cosines and
eta-kappa use complete 100-pair trailing means; loss, norms, ratios and B/A
are unsmoothed. Only panels (h) and (i) use log Y. Colors encode LR consistently.

Sources: frozen Runs063-066 production `metrics.jsonl`, `curvature.jsonl`,
`summary.json` and final evaluations, the [cohort](../cohort.json),
[results](../results.json), and the source/derived-file hashes in
[verification](../verification.json). The verified `derived/Run063.npz`
through `Run066.npz` are checked by size and SHA256 before plotting; complete
scalar histories and NPZ arrays remain locally outside Git. See Observation001
for the unchanged data, loss, initialization and Adam configuration.

Momentum definition: mhat_t = m_t/(1-beta1^t), beta1=0.9, the bias-corrected
first moment. It excludes second-moment division and is distinct from Adam's
adaptive direction or actual parameter update. Panel (c) uses
cos(mhat_(t-1), mhat_t), rather than alignment of mhat_t with g_t. Panel (g)
uses ||sum_(s=1)^t mhat_s|| / sum_(s=1)^t ||mhat_s||. Bias correction does not
change adjacent cosine under positive scaling, but it changes early norms and
cumulative B/A relative to stored m.

Norm ratio: panel (j) computes ||g_t|| / ||mhat_t|| directly at matching update
t, before any averaging. This is also independently checked as
(1-beta1^t)/(||m_t||/||g_t||) against the recorded stored-m/raw ratio. It is
neither mhat/raw nor the uncorrected raw/stored-m ratio. All 80,000 measured
denominators are positive; zero-denominator values would be undefined.

Kappa reduction: eta*kappa_hat_t = 1-rho_t*||g_t||/||g_(t-1)||, with explicit
noise term zero, is computed before its mean. All three mean panels use exactly
the last 100 defined pairs, right-aligned at update t. The first displayed mean
is at update 101. There is no partial window, zero padding, factor-wise mean,
or thinning. The zoom selects the same already-computed series, without
restarting cumulative sums or rolling windows. The Hessian panel uses ordinary
lambda_max, with no preconditioning, division by 2, or GD threshold line.

Coverage: four Adam rates, one matched frozen seed, all 656,810 parameters and
all 20,000 updates. Full loss covers states 0..19,999 and the separately evaluated
state 20,000; full gradient/momentum panels cover updates 1..20,000. The early
loss covers states 0..2,000, while early gradient/momentum panels cover updates
1..2,000. Raw g_t belongs to theta[t-1], mhat_t is updated using that gradient,
and Hessians belong to their explicitly named states. Ordinary Hessians cover
201 full-horizon states or 21 early states, measured every 100 updates. Each
mean panel displays 19,900 means per LR in the full view and 1,900 in the zoom.

Visual reductions: the two figures have identical Y limits for each metric.
Raw/momentum norms also share a common log range within each figure. Loss
remains linear, with a range that preserves the highest early Adam loss spike;
its Y range differs from the GD figure for this reason. Ordinary eta*lambda_max
uses a smaller linear range than GD. The ratio remains linear and includes all
observed spikes. No Y-range clipping of any displayed curve is allowed.

Observation: the early view separates the raw cosine reversal from the later
momentum cosine reversal. Momentum B/A remains larger than raw B/A at the
fixed horizon. The raw-to-momentum norm ratio is generally much larger than
one after the early transient. Meanwhile eta-kappa is near 1.8-1.9 late in
training and ordinary eta*lambda_max remains below 0.1. These descriptive
patterns do not make the gradient proxy an ordinary-Hessian estimate on Adam.

Uncertainty and caveats: one seed, finite horizon, no confidence bands or tuned
LR comparison. Curvature solves use the full 5k objective and pass relative
eigenpair residual <= 0.005 but have only 100-step temporal resolution.
The raw-gradient kappa expression is a trajectory proxy along adaptive Adam
updates; it is not the plain-GD directional curvature identity on this path.
Momentum similarity, B/A and norm ratios do not alone establish stability,
loss decrease or generalization. Scalar histories do not recover missing dense
long-lag vectors or per-example gradient decompositions.

Generator: [10_plot_adam.py](../10_plot_adam.py), reusing the checked window
and hashing helpers in [09_plot_gd.py](../09_plot_gd.py).
[adam_figure_manifest.json](../adam_figure_manifest.json) records all 80 plotted
arrays, 238,800 unique complete means checked against independent direct
windows, and 261,600 displayed mean values across both views. It also checks
mhat scaling, both B/A ratios, kappa arithmetic, the norm-ratio direction and
curve bounds. The GD PDF and archive manifest hashes remain unchanged.
[07_pdf_qa.py](../07_pdf_qa.py) with `--simple-adam` renders and checks both
PDFs; [adam_visual_verification.json](../adam_visual_verification.json)
records the inspected render and page/text-bound identities.
