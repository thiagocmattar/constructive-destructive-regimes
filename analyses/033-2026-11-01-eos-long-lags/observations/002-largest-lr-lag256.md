# Observation 002: largest GD rate with consecutive lags connected

Requested view: one learning rate per architecture, its largest measured GD
rate, and lags 1 through 256 without separating odd from even lags.

Sources: the existing `results.json` and independently exported `lag_profiles.csv`.
MLP uses Run122 at LR 0.04, ResNet-32 uses Run120 at LR 0.1, and Transformer
uses Run126 at LR 0.2. All nine selected profiles are primary, with retained
anchors at 500, 2000 and 3000. No new training or gradient measurement is needed.
Source hashes, generator identity, plotted-value hashes and checks are in
`largest_lr_lag256_review.json`.

Reduction: select those nine profiles and retain all 256 positive lags from each.
Each panel contains one line joining the exact recorded cosine values in lag
order, for 2304 points total. No parity split, smoothing, averaging, projection,
downsampling or envelope is used. The linear axes share x=1..256 and y=-1.05..1.05.
Every artist's values are checked against the JSON source and the CSV export.
The source's full-objective gradients, parameter scope and FP32/FP64 precision
contract remain those of Observation001.

Caption, `figures/largest_lr_lag256.pdf`: Whole-model raw-gradient cosine
cos(g₀,gₖ) at lags 1–256 along GD continuations from retained states. Here g₀ is
the gradient at the row's anchor, rather than at training initialization.
Columns show MLP (LR 0.04), ResNet-32 (LR 0.1), and Transformer (LR 0.2);
rows show anchors 500, 2000 and 3000. All consecutive lags are connected as a
single curve in each panel, exposing the raw alternating zigzag. The horizontal
line marks zero. This is an anchored cosine, not an adjacent-gradient cosine.

Coverage and uncertainty: nine dependent profiles, one initialization per model,
256 anchored pairs per profile, no uncertainty bands. The two unqualified
Run062 reconstructions are outside this largest-rate subset. Rapid alternation
can appear as a dense band when all points are connected. The figure adds a
display of existing evidence; it does not establish permanent damping,
optimization stability, generalization, or a causal cross-architecture effect.

Generator: `12_plot_largest_lr_lag256.py`. Publication output is the vector PDF;
the actual-PDF preview is retained under ignored `scratch/largest_lr_lag256.png`.
