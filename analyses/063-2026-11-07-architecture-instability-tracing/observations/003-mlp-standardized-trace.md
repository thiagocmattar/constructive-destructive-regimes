# MLP: the standardized architecture trace

## TLDR

Figure 03 puts the existing MLP measurements into the same two-page layout as
the ResNet and Transformer reports. It covers every learned representation and
all 5,000 training examples over outgoing updates 448-800. No new training,
data sampling, optimizer intervention or cloud execution was required.

The figure preserves the earlier finding: reversal is already present at the
first linear projection and closely follows the output throughout the network.
The first-linear/output alignment correlation is 0.9910 and their top-ten
temporal overlap is 0.9956. Top-ten modes retain 95.57% of first-linear change
energy and 98.16% of output-change energy. These locate observed oscillation;
they do not establish a unique causal initiating layer.

## Coverage, quantities and comparison contract

Source measurements are the exactly qualified seed-0 Adam continuation in
Runs 240-242, analyzed previously in Analysis 062. Architecture: flattened
3072-dimensional CIFAR-10 images, Linear(3072,200), tanh, Linear(200,200), tanh,
Linear(200,10). Adam LR 0.001, betas (0.9,0.999), no clipping or decay; physical
batch size 1,000. The original objective uses the fixed 5,000 training examples.

For raw output matrix F and one-hot targets Y, the full objective is

$$L_t=\frac{1}{2\cdot5000}\sum_{i=1}^{5000}\|F_{t,i}-Y_i\|_2^2.$$

The diagnostic set equals this entire training set. Therefore page 1 shows one
loss curve labeled **Full objective = diagnostic set**, rather than inventing a
smaller probe. Outputs are raw linear values, matching the squared-error loss.
No log-softmax or correct-label log-probability diagnostic is introduced. The
other two architecture reports use log-probabilities and smaller fixed probes;
their coverage and output spaces remain explicitly different.

At state t, t optimizer updates have completed. The displayed movement is
Delta A_t = A_(t+1) - A_t, and its cosine compares Delta A_t with Delta A_(t-1).
Captured states 447-801 supply every endpoint. Raw-gradient alignment remains
cos(g_t,g_(t-1)); gradients are not differenced.

Each hidden/preactivation representation has shape 5,000 x 200, hence P=1,000,000
coordinates and a 353 x 1,000,000 displacement matrix. The final output has shape
5,000 x 10, P=50,000 and a 353 x 50,000 displacement matrix. The raw-gradient
stream has 656,810 parameters. Fixed inputs have exactly zero displacement;
their motion cosine is undefined and no artificial heatmap row is assigned.

## Figure caption and reductions

**Page 1:** full-objective loss; full raw-gradient and raw-output consecutive
alignment; five-row representation-change cosine heatmap; displacement coordinate
RMS divided by each row's own window maximum; top-one/top-ten singular energy
fractions and top-ten temporal-subspace overlap with raw output. The rows are
Z1, H1, Z2, H2 and final output, in forward order.

**Page 2:** every learned representation, with consecutive-change cosine,
absolute displacement coordinate RMS, and leading fixed SVD coefficient divided
by sqrt(P). RMS is ||Delta A_t||_F/sqrt(P); only the overview heatmap divides by
the stage's window maximum. These normalizations do not make different layers'
natural scales or units identical.

The [common method](../METHOD.md) uses exact FP64 temporal Grams of retained
FP32 measurements, without sample averaging, temporal centering or row
normalization. Each basis is fitted once over all 353 updates. The sign is fixed
once by making the largest absolute temporal coefficient positive, matching the
ResNet/Transformer report convention. Compared with the original MLP spatial
sign convention, the displayed leading coefficients are globally negated for
each stream; their magnitudes and reversal times are unchanged. Independent
layer bases do not identify identical spatial directions.

## Sources, reproduction and verification

- [Figure 03](../figures/03-mlp-instability-trace.pdf).
- [Complete numerical results and source hashes](../mlp_results.json).
- [Stage summary](../mlp_stages.csv).
- [Original all-layer observation](../../062-2026-11-07-mlp-gradient-amplification/observations/012-all-layers-and-fixed-input.md).
- [Original gradient/hidden/output observation](../../062-2026-11-07-mlp-gradient-amplification/observations/010-gradient-output-hidden-svd-results.md).
- [Numerical figure audit](../mlp_figure_verification.json) and
  [render/text/retention audit](../mlp_visual_verification.json).

Run `06_mlp_trace.py`, then `02_figures.py --case mlp` in this analysis. The adapter
checks source CSV identities, the earlier audited Gram hashes, fixed-input
verification, identical time/loss rows, and identical output Grams across source
analyses. For all six streams, all 353 cosines, norms, singular energies and
leading coefficients agree with the earlier results (coefficients up to one
fixed sign). The maximum leading-coefficient discrepancy is below 9e-14.
The first-linear/output temporal overlap also matches independently. Both PDF
pages were rendered and visually reviewed, and all text lies inside the pages.

Coverage is one selected seed and window, with no new validation or generalization
measurement. The larger 353-update MLP window differs from the 31/49-update
ResNet/Transformer windows, so top-k concentration is not a matched comparison
of intrinsic dimensionality. No causal intervention is implied by this figure.
