# Mean network outputs across updates, with loss and alignment

Historical version: the user subsequently requested SVD patterns in the same
PDF. The current figure and generator are documented in
[observation006](006-svd-output-change-heatmap.md). This note and its mean-output
CSV preserve the preceding reduction; its implementation is retained in Git
commit257620d. References to the mean heatmap below describe that earlier version.

The user requested a tall output heatmap beneath the full loss curve and the
temporal gradient/output-change alignment overlay. On clarification, the user
selected **10 rows: mean output across examples**. This figure uses that
aggregation, not one example or50,000 individual coordinates. It uses retained
Run239 outputs; no training or remote execution is required.

Figure: [Loss, alignments and mean-output heatmap](../figures/05-network-output-heatmap.pdf).
Generator: [06_output_heatmap.py](../06_output_heatmap.py), default
`--representation mean`.
Mean values: [mean_output_heatmap_values.csv](../mean_output_heatmap_values.csv).
Audit: [output_heatmap_verification.json](../output_heatmap_verification.json).

## Definition and caption

Each heatmap column is a completed optimizer state $t=448,\ldots,800$. Each row
is one output coordinate $c=0,\ldots,9$ in CIFAR10 class order. The cell value is

$$
H_{c,t}=\frac1{5000}\sum_{i=1}^{5000} f_{\theta_t}(x_i)_c.
$$

The average uses all5000 fixed training examples for every row, irrespective of
their true labels. Row names denote the output coordinates, not subsets of
examples. These are raw final linear-layer values, not softmax probabilities,
classification frequencies, residuals or output differences. The mean is
calculated in FP64 from the retained FP32 outputs. There is no per-row
normalization, centering, sorting, smoothing, temporal subsampling or value
clipping. A single shared sequential color scale spans the minimum and maximum
of all3530 mean values; it is not anchored at zero.

**Caption. Unchanged CIFAR-10 MLP Adam control, states448--800.** (a)Full training
loss $L_t$, using the original half-squared-error objective over all5000 examples.
(b)Overlay of consecutive raw-gradient alignment and consecutive output-change
alignment, with exactly the same values and indexing as Figure04.
(c)Mean raw network output for each of the10 output dimensions, averaged over
all5000 training examples at each state. All three panels share the same time
axis. The heatmap is taller than the line panels, and its colorbar uses one
common value scale. The complete10-by353 matrix is embedded without interpolation.

The overlay definitions remain

$$
\rho_t=\cos(g_t,g_{t-1}),\qquad
\rho_t^{(\Delta F)}=\cos(F_{t+1}-F_t,F_t-F_{t-1}),
$$

where $F_t$ concatenates all50,000 raw outputs. Thus the output-alignment curve
uses the complete per-example output space, whereas the heatmap displays its
10 coordinate-wise means. It does not recompute alignment from the10 means.

## Coverage, sources and verification

Same seed0 CIFAR10 MLP, fixed examples, unchanged Adam control, precision and
state convention as [observation003](003-control-coherence-448-800.md). Full loss
is read from Run239 `boundaries.jsonl`. Outputs come from Run239 `outputs.pt`,
which covers447..801; this figure selects448..800 inclusive. Both alignment
series are read from the previously verified `control_coherence_448_800.csv`.
Source hashes are checked against `control_coherence_verification.json`.

The generator checks all3530 heatmap values against the Matplotlib image array
and all1059 line values against plotted artists. Independent Torch FP64 means
are checked against the saved mean-output CSV. The rendered PDF is reviewed
for row labels, common time-axis placement, colorbar, formulas and clipping;
its embedded heatmap dimensions are checked as353 columns by10 rows. Source,
generator, CSV and final PDF hashes are saved in the audit. Raw arrays remain
retained in their owning run; no scientific inputs are modified.

## Interpretation limits

The preceding figure's Pearson correlation0.776 is a strong descriptive
association, with visible departures from equality; it is not near-perfect
agreement or an independent-sample statistical claim.

Positive and negative example-specific output changes can cancel in these
means. Consequently, small changes or weak visual bands in this heatmap do not
establish that individual predictions or representations stay fixed. Conversely,
visible mean changes alone do not explain the full squared-error loss, which
depends on individual residuals. The figure is descriptive for one seed and
trajectory; it contains no test-set inference or causal intervention.
