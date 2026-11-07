# Fixed SVD patterns of output changes, retaining example structure

The user requested replacing Figure05's per-class mean heatmap with SVD
patterns, and explicitly selected retaining individual-example structure.
The current PDF implements a fixed full-example SVD basis for the complete
448--800 window. The loss and temporal-alignment curves remain unchanged.
This is a post hoc reduction of retained Run239 outputs, with no new training.

Figure: [Loss, alignment and SVD-pattern heatmap](../figures/05-network-output-heatmap.pdf).
Generator: [06_output_heatmap.py](../06_output_heatmap.py).
Coefficients: [svd_output_heatmap_values.csv](../svd_output_heatmap_values.csv).
Audit: [output_heatmap_verification.json](../output_heatmap_verification.json).
Independent check: [07_verify_svd_heatmap.py](../07_verify_svd_heatmap.py) and
[verification result](../svd_heatmap_independent_verification.json).

## Matrix, basis and heatmap

For each $t=448,\ldots,800$, concatenate all10 raw network outputs for all5000
fixed examples in the same order. Let $F_t\in\mathbb R^{50000}$ be this vector.
Build the matrix

$$
D\in\mathbb R^{353\times50000},\qquad D_{t,:}=F_{t+1}-F_t.
$$

The source includes state801 to supply the outgoing change at800. Neither
examples nor output coordinates are averaged. No row/column centering or
standardization is applied: the decomposition measures squared displacement
energy, not centered variance. Compute an uncentered SVD via the temporal Gram
matrix $DD^\top$, in FP64:

$$
D=U\Sigma V^\top,\qquad a_{t,k}=(U\Sigma)_{t,k}=D_{t,:}v_k.
$$

The first10 right singular vectors $v_k\in\mathbb R^{50000}$ are fixed for
the entire window, ordered by decreasing squared singular value. Each can be
reshaped to5000 examples by10 outputs. The top10 rows retain the prior figure's
10-row layout while exposing the major full-example patterns.

The heatmap cell is

$$
H_{k,t}=\frac{a_{t,k}}{\sqrt{5000}},\qquad
D_{t,:}\approx\sum_{k=1}^{10}a_{t,k}v_k^\top.
$$

The common division by $\sqrt{5000}$ expresses each mode's magnitude on a
per-example vector-RMS scale; it does not average examples or normalize each
mode separately. Because $\|v_k\|=1$, the absolute heatmap value equals the
RMS norm across examples of that mode's output-change contribution.

SVD signs are arbitrary. For reproducibility, orient each fixed pattern so its
largest-absolute coordinate is positive, and apply the same sign change to its
coefficients. The sign is chosen once per pattern, never separately per state.
Red/blue therefore indicate opposite coefficients of the same pattern, and
alternating colors reveal temporal reversals. Zero is white. A single symmetric
color scale covers all cells, with no clipping or per-row color normalization.

## Energy coverage and what the rows mean

The row annotation is $\sigma_k^2/\|D\|_F^2$, measured over the full window:

| Pattern | Change-energy share |
|---|---:|
| 1 | 25.17% |
| 2 | 21.24% |
| 3 | 17.03% |
| 4 | 13.81% |
| 5 | 8.52% |
| 6 | 4.15% |
| 7 | 3.09% |
| 8 | 2.65% |
| 9 | 1.46% |
| 10 | 1.04% |

The first10 patterns capture98.1569% of total squared output-change energy;
the omitted patterns account for1.8431%. These are signed temporal coefficients,
not singular values recomputed at each step. The fixed basis prevents arbitrary
sign flips or pattern swaps between adjacent columns. The stored patterns
preserve which examples and output coordinates contribute.

This is **different from the earlier93.4% finding**. That finding decomposed a
5000-by10 matrix of before-to-peak changes for one selected event, yielding a
shared10-dimensional class-space direction and example-dependent amplitudes.
Here a353-by50,000 matrix spans many updates and uses fixed patterns containing
both example and class structure. Multiple full-window patterns are compatible
with a locally near-rank-one class-space change whose direction varies over
time. The25.17% first-mode share does not contradict the earlier93.4% result.

## Caption

**Output-change SVD patterns in the unchanged CIFAR-10 MLP Adam control.**
(a)Full training loss and(b)the unchanged temporal gradient/output-change
alignment overlay over states448--800. (c)Signed coefficients of the first10
fixed full-example output-change SVD patterns, divided by sqrt(5000). Each row
is one50,000-coordinate pattern, and each column is the outgoing update at$t$.
Row labels give full-window energy shares; the ten modes capture98.16% of
total change energy. Red and blue correspond to opposite coefficients under
a fixed sign convention. The basis uses all examples without averaging. The
complete10-by353 coefficient matrix is embedded in the PDF without interpolation.

## Sources, verification and limitations

Same pinned Run239 outputs and native boundary losses as the preceding figure;
same verified CSV for both alignment curves. Source SHA256 checks remain active.
All3530 heatmap values and1059 line values are checked against plotted artists.
The decomposition verifies orthonormality, direct projections and residual
energy. `07_verify_svd_heatmap.py` independently projects the raw FP64 output
differences with NumPy, checks the exported CSV, verifies the deterministic
signs and energy shares, and reconstructs the full displacement matrix in
blocks to verify omitted energy. Results and source/PDF hashes are retained.

The basis and coefficients are retained in the ignored local
`artifacts/svd-output-changes/patterns.pt`; it is reproducible from the pinned
Run239 raw outputs. The previous mean-output CSV and observation005 remain as
historical records; the PDF at the requested path is replaced in place.

This basis is fitted post hoc to the entire window, including later states.
It is descriptive, not an online predictor or a training intervention. High
energy coverage does not imply an equivalent fraction of loss variation is
explained. Pattern signs are conventional, and near-degenerate singular
subspaces can admit alternative bases. One seed/window gives no cross-seed
uncertainty interval. The layout and temporal indexing do not establish causality.
