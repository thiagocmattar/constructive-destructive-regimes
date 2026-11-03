# Pre-rise cosine times rise gradient norm ratio versus peak loss ratio

Figure: [adam_rho_gradient_rise_peak_scatter.pdf](../figures/adam_rho_gradient_rise_peak_scatter.pdf).
Generator: [11_plot_rho_gradient_rise_peak_scatter.py](../11_plot_rho_gradient_rise_peak_scatter.py).
Source hashes, event IDs, exact plotted coordinates and checks:
[manifest](../rho_gradient_rise_peak_scatter_manifest.json).

Caption: Peak/pre-rise loss versus the product of pre-rise consecutive raw-gradient
cosine and rise gradient norm ratio, under controlled full-batch Adam. Columns show
CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer; colors identify constant
LRs within each model. The x-coordinate is rho_(a+1) * ||g_(p+1)|| / ||g_(a+1)||.
The cosine uses gradients at states a-1 and a; the norm ratio compares the loss-peak
state p with the last pre-rise state a. The product retains its sign, without
negation or absolute value. The shared x-axis is linear (-16..2.5).
The shared logarithmic y-axis (1..11) is loss_peak / loss_before, with 1 meaning
no increase, 2 meaning +100%, and 10 meaning +900%. All points are unsmoothed;
no averaging, jitter or regression. Layout, typography and LR colors match the
existing 7.16 x 2.55 inch Analysis024-style figures.

Sources and coverage: current anatomy events with peak increase >10% and observed
recovery by update 1024, from existing Run215 histories. Two initialization events
have undefined pre-rise cosine and are omitted; the manifest records their IDs.
127 points remain: ascending-LR counts MLP [8,46,38], ResNet [0,10,7],
Transformer [0,6,12]. Earlier scatter inputs and figures are preserved.

Observation and limits: 117 of 127 products are negative, reflecting the sign of
the pre-rise cosine. Larger loss ratios tend toward more negative products in
parts of the scatter, with substantial dispersion. Visual straightness on this
semilog plot concerns log(loss ratio) versus x, not linear loss ratio versus x.
No fit or correlation estimate is introduced. This mixes a pre-rise cosine with
a retrospective multi-update norm ratio, so it is neither a prospective predictor
nor the pointwise eta*kappa_hat expression (which uses a consecutive norm ratio).
The two axes share loss-defined event endpoints. Selection on future amplitude
and recovery, temporal dependence within single-initialization trajectories,
and model/LR differences limit interpretation. No causal claim, new training,
cloud execution, intervention or manuscript change.

Verification: source SHA-256, amplitude/recovery criteria, unique IDs, missing
initialization cosines, cosine bounds, endpoint norm ratio, signed product and
loss-ratio identities checked. Artist coordinates match computed values exactly;
all points are inside the common axes. One vector PDF page, embedded fonts and
text within page bounds. Poppler render visually reviewed: readable panels,
legends and labels, with no clipping or overlap.
