# Observation004: smoothed model-size overlays and early loss reduction

The user requests the four training variables in a 2x2 layout, overlaying all
three Pythia sizes and removing the final step. The latest revision keeps loss
unsmoothed, smooths the other variables, and uses solid lines in three colours. The
[new figure](../figures/training_size_overlay.pdf) uses Runs017/018/019 and seed
1235. No training or new run/analysis ID is needed.

## Sources, filtering and reduction

Sources and matched factors are those in [observation003](003-training-overview.md):
the actual Analysis005 measurement replays, with identical MiniPile cache/order,
token coverage, accumulation, AdamW recipe, LR schedule, precision and raw-gradient
capture across model sizes. Architecture and its initialized tensors vary. Loss
is the changing minibatch's mean supervised-token cross-entropy. Raw g includes
all unique trainable parameters before clipping or AdamW transformation.

The generator checks the preceding overview's CSV hash and the sizes/hashes of
all underlying consumed run sources. Update 712, the partial 311-sequence batch,
is excluded BEFORE every moving mean and summary. Each size contributes updates
1..711, all full 1024-sequence batches. The original complete histories and PDFs
remain unchanged.

Loss is unsmoothed and covers updates 1..711. Each other plotted quantity uses
an arithmetic mean over its own last 20 raw values, including the current update.
Only complete windows are shown: norm begins at update 20; rho and q begin at
21 because the first pair is undefined.
All curves end at 711. There are no shortened edge windows, interpolation,
subsampling, raw underlays or uncertainty bands. A trailing mean averages prior
states and can delay or blur visible transitions by up to 19 updates.

Compute q=1-rho_t*norm(g_t)/norm(g_(t-1)) at every raw pair BEFORE smoothing.
The plotted q is the mean of those products, not a product of smoothed rho and
norms. The raw q>1 iff rho<0 identity does not imply the same sign equivalence
for the separately smoothed panels. Gradient norms are arithmetically averaged
before displaying them on log Y, not averaged in logarithmic coordinates.

The script verifies all 2,130 retained raw identities and independently recomputes
every complete window by slicing (6,222 means), with all 8,355 finite plotted values checked
against their artists and the exported CSV. Missing means remain blank in the
CSV and NaN in the plotted arrays. Poppler rendering and visual inspection are
recorded in the [manifest](../training_size_overlay_manifest.json). All 2,133
unsmoothed loss values are checked against their source values; all model curves
are checked to use solid lines and their assigned colours. The CSV column is
`loss` for the raw loss; diagnostic columns retain the `_mean` suffix.

## Early loss comparison

The user's possible explanation is faster initial loss reduction at larger size,
associated with higher raw-gradient norms. The following summaries use the
UNSMOOTHED retained values. These are exploratory window summaries, not a
pre-registered hypothesis test. Endpoint reductions and the average per-update
decrease use L1-LN and (L1-LN)/(N-1), respectively. Norms are arithmetic means
over updates 1..N. Initial losses are 11.0230/11.0073/10.9828 in increasing-size order.

| Model | Loss drop, 1..10 | Mean norm, 1..10 | Loss drop, 1..50 | Mean norm, 1..50 | Loss at 711 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14M | 1.0081 | 1.1500 | 3.5259 | 0.7593 | 5.2328 |
| 31M | 1.7561 | 1.5253 | 4.1735 | 0.6863 | 4.6154 |
| 70M | 2.6316 | 1.7909 | 4.4768 | 0.7724 | 3.9663 |

Larger models make larger cumulative loss reductions over the first 10, 50 and 100
updates. Both mean raw norm and early loss drop increase with size in the first 10.
The norm ordering already fails over 1..50: 31M has a smaller mean norm than 14M,
despite its larger loss decrease. First-100 means are 0.5500/0.5087/0.5650 and show
the same non-monotonicity. Beyond the initial transient, the norm overlays are
close and often cross, while the loss curves remain separated.

"Steeper" depends on the summary and window. The fitted ordinary least-squares
loss slopes over 1..50 are -0.07230/-0.07709/-0.07082 per update; the 70M fit is not
the steepest because much of its advantage appears in the first few updates.
The endpoint-average decreases over that same window are 0.07196/0.08517/0.09136.
Both summaries are retained in [results](../training_size_overlay_results.json).
Do not infer a larger instantaneous derivative throughout training from the
larger cumulative decrease.

The raw first-negative pairs remain at updates 63/26/14 in size order, predating
the displayed trailing-mean crossings. The 70M rho recovers less positively late
in training and its q remains closer to 1, despite its lower loss. These patterns
describe these trajectories and do not establish that either metric causes
faster learning or measures stability.

## Why loss slope does not determine the raw norm here

For the SAME smooth objective F, a small plain-GD step delta=-eta*g gives
F(theta+delta)-F(theta) approximately -eta*norm(g)^2, to first order. More
generally, the first-order quantity is grad(F)^T delta, so displacement magnitude
and direction relative to the gradient matter.

These runs use clipped, momentum-based, diagonally preconditioned AdamW updates,
decoupled weight decay and a changing LR. Their actual displacement is not
-eta times the recorded raw g. Consecutive training losses also evaluate different
minibatches, so their difference is not even a same-objective finite difference.
Consequently, higher raw norm is not a mathematical consequence of steeper
observed loss reduction here. The first-10 association is descriptive, not a
mechanistic attribution. Global L2 norms also depend on parameter count and
parameterization; these measurements do not normalize that dependence away.

There is one seed per model size, without seed uncertainty or a causal size
mechanism test. The AdamW/minibatch q is a dimensionless gradient-change proxy,
without a measured noise correction or verified Hessian interpretation.

## Caption and reproduction

**Caption.** Pythia-14M/31M/70M MiniPile training dynamics (Runs017-019, seed 1235).
Panels show training loss, eta*kappa_hat gradient-change proxy, raw adjacent
cosine rho and raw gradient L2 norm, with all model sizes overlaid. Loss is
unsmoothed over updates 1..711; other curves are complete 20-update trailing
arithmetic means computed after omitting update 712. Norm means cover updates
20..711; pair means cover 21..711. Solid blue, orange and green lines denote
14M, 31M and 70M. The norm axis is logarithmic;
horizontal dotted references mark q=1 and rho=0. These are descriptive
trajectories, without confidence bands or curvature/stability implications.

Run `python analyses/005-2026-10-04-temporal-lag-profiles/13_plot_size_overlay.py`.
Outputs are the PDF, [plot data](../training_size_overlay_data.csv),
results and manifest. Render using `pdftoppm -png -r 130` to
`scratch/training_size_overlay`, inspect the page, then run the script with
`--accept-reviewed`. Original unsmoothed observations and figures are preserved.
