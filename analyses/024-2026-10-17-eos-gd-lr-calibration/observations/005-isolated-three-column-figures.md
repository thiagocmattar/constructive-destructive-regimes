# Observation005: isolated three-column training-dynamics figures

Question: how do the completed Analysis024 trajectories read when each
diagnostic occupies an isolated one-row, three-column figure?

## Sources and coverage

All nine conditions from `cohort.json` and the independently verified
`results.json`: MLP LRs0.004/0.02/0.04 (Runs059/062/122), ResNet-32
LRs0.025/0.05/0.1 (Runs123/124/120), and Transformer LRs0.05/0.1/0.2
(Runs121/125/126). Each has3,000 constant-LR full-GD updates and seed0,
with a matched frozen initialization within each model case. The objectives
cover5k CIFAR-10 images,50k CIFAR-10 images with fixed ghost-BN groups, and
all2,088,600 prepared WikiText-2 target tokens per update, respectively.

The qualified Analysis020 loader reads the owning runs' original metrics,
summary, manifest and spectra. Used metrics/spectra/manifest SHA256 hashes
must match the independent Analysis024 reduction; configs and frozen initial
tensor identities are also checked. The final loss and R_t endpoint match
the existing reduction exactly. Raw evidence is unchanged; no training,
gradient reconstruction, new sampling or remote execution is involved.

## Reduction and indexing

Generator: `../07_plot_single_rows.py`. Compact results/provenance:
`../single_rows_results.json`, `../single_rows_figure_manifest.json`.
Exact plotted arrays: ignored `../derived/single_rows_curves.npz`.

Smoothing is the unweighted arithmetic mean of50 consecutive observations,
aligned to the last observation in each complete window. There is no padding,
partial-window mean, temporal subsampling or seed averaging. The mean of the
pointwise q observable is plotted; it is not q recomputed from smoothed inputs.
Gradient norms are averaged in their original units before log display.

Loss is indexed by parameter states0..3000 and includes the measured endpoint
(the next preupdate loss row for reused MLP prefixes). Norms use evaluation
states0..2999. Consecutive cosine and q use evaluation states1..2999; the
undefined first pair is omitted. Their first complete50-pair mean is at
state50; the first complete50-norm mean is at state49. R_t=B_t/A_t is the
cumulative statistic after updates1..3000, without smoothing or window reset.
All panels use the horizontal label 'GD iteration'; this one-index difference
between preupdate gradient states and completed-update prefixes is explicit.

No nonpositive value is floored or clipped for logarithmic display. Cosine
uses the common range[-1.06,1.06]; R_t uses[-0.025,1.025]. Other Y limits
are independent across model cases because their objective scales differ.

## Outputs and captions

All PDFs are in `../figures/`; matching Poppler PNGs are in ignored
`../scratch/`. Each is one row by three columns, ordered MLP / ResNet-32 /
Transformer. Within each column, blue / orange / green denote increasing LR;
all data curves are solid, as requested. Legends give the actual model-specific
rates. Dashed horizontal lines denote reference values only.

| PDF | Metric / caption detail | Y scale | Points across nine curves |
| --- | --- | --- | ---: |
| `eos_gd_loss_log_3k.pdf` | Unsmoothed full-objective training loss, including state3000 | Log | 27,009 |
| `eos_gd_rho_3k.pdf` | Unsmoothed consecutive-gradient cosine; dashed zero reference | Linear | 26,991 |
| `eos_gd_rho_smoothed_3k.pdf` | Complete50-pair trailing mean cosine; dashed zero reference | Linear | 26,550 |
| `eos_gd_eta_kappa_hat_smoothed_3k.pdf` | Complete50-pair trailing mean of q=eta*kappa_hat; dashed q=1 reference | Linear | 26,550 |
| `eos_gd_gradient_norms_smoothed_3k.pdf` | Complete50-gradient trailing mean raw-gradient norm | Log | 26,559 |
| `eos_gd_normalized_displacement_3k.pdf` | Unsmoothed cumulative R_t=B_t/A_t after every update | Linear | 27,000 |

Shared caption: completed full-batch GD trajectories through3,000 updates,
with three constant learning rates per model/task case and one matched
initialization. Each footer identifies seed, update coverage and smoothing.
The54 curves contain160,659 plotted points. No uncertainty band is shown
because there is one initialization per condition.

## Observations and limits

The isolated views make sustained directional reversal and the subsequent
decline of normalized displacement easier to compare. The higher-MLP-rate
initial transient remains visible in the unsmoothed view; complete trailing
means delay and broaden abrupt transitions. Do not read a smoothed boundary
crossing as the original first negative pair or as a newly defined onset.

These views reproduce the previously recorded findings: late negative
alignment and low cumulative R_t can coexist with continued loss reduction.
R_t is path-relative accumulation, not an absolute displacement or performance
measure. The q/cosine relationship is algebraic; agreement is not independent
curvature validation. No Hessian curve appears in these six figures, and q
is not maximum Hessian curvature. A local curvature interpretation retains
the Taylor approximation. No global-stability, convergence, generalization,
initialization-robustness or prospective-warning claim is established.

## Verification

All used-source identities and complete nine-condition coverage pass. The
q identity, cumulative A=sum(norms), R_t=B/A and recorded endpoint checks
pass. Every smoothed value agrees with an independent convolution, every
Matplotlib artist array agrees with its source/reduction, and all log-axis
values are positive. Each PDF has one page with the expected model/coverage
text, and all figure bounds lie within the canvas. All six current Poppler
renders were inspected for legends, labels, axes, reference lines and clipping.
The accepted figure manifest pins script, loader, cohort, arrays, results,
PDF and render hashes. Original overview PDFs and source evidence are preserved.
