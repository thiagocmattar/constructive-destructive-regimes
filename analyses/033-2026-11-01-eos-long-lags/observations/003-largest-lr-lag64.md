# Observation 003: largest-rate GD figure through lag 64

Sources: the same nine primary profiles as Observation002, selected from
`results.json` at MLP LR 0.04 (Run122), ResNet-32 LR 0.1 (Run120), and Transformer
LR 0.2 (Run126), at anchors 500, 2000 and 3000. Source JSON and CSV hashes must
match the verified 256-lag view. No new training or gradient measurement.

Reduction and coverage: retain every lag 1..64, 64 anchored pairs per profile,
576 points total. Each panel joins all consecutive lags in one line. No parity
split, smoothing, averaging, projection or downsampling. Every plotted artist
is checked against its source arrays; the previous 256-lag review independently
checked these same values against the CSV. Linear x limits are 1..64, and the
layout, rates, anchors and cosine y limits match the 256-lag view.

Caption, `figures/largest_lr_lag64.pdf`: Whole-model raw-gradient cosine
cos(g₀,gₖ) through lag 64 at the largest GD rate per architecture. Columns show
MLP (LR 0.04), ResNet-32 (LR 0.1) and Transformer (LR 0.2); rows show retained
anchors 500, 2000 and 3000. Here g₀ is the gradient at the row's anchor.
All lags are connected in order, with a horizontal zero reference.

Uncertainty and caveats: one initialization per architecture, dependent anchor
forks and lag pairs, no uncertainty bands. This shorter view omits later
rebounds visible in the 256-lag figure. Anchored cosine is distinct from
adjacent-gradient cosine and does not establish stability or generalization.
Objective coverage and FP32-gradient/FP64-cosine precision are inherited from
Observation001; the two unqualified Run062 reconstructions are outside this subset.

Generator: `13_plot_largest_lr_lag64.py`. Source identities, 576-point checks,
PDF identity and visual inspection are recorded in `largest_lr_lag64_review.json`.
The vector PDF is the publication output; its actual-PDF preview is ignored
under `scratch/largest_lr_lag64.png`.
