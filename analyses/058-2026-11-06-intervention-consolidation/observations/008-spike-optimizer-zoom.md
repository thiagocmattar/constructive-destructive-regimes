# 08: Spike zoom with current momentum alignment and norm ratios

Requested 7 November 2026: recreate the manuscript's
`manuscript/draft/figures/analysis036/adam_manuscript_spikes.pdf` in Analysis058,
adding current gradient/momentum cosine and momentum-to-gradient norm ratios.
Generator and checks: `../05_spike_optimizer_zoom.py`. Sources, hashes, exact
curves and check counts: `../spike_optimizer_zoom_manifest.json`; plotted table:
`../spike_optimizer_zoom_values.csv`. No training, cloud spending or manuscript edit.

## Sources and preserved scope

The manuscript PDF is byte-identical to Analysis036's retained spike PDF at
generation time. Its `01_plot_spikes.py`, archived full unsmoothed curve manifest,
paper-evidence manifest and selected-example records define the first four rows.
The added diagnostics come directly from the same authenticated Run215 metrics.
The original 18 curves / 1,056 values are preserved, including separate raw and
momentum adjacent cosines and norms relative to each stream's crop-start value.

| Model | Constant LR | Inclusive states | Historical onset marker | Peak |
| --- | ---: | --- | ---: | ---: |
| CIFAR-10 / MLP | 0.001 | 693--729 | 697 | 703 |
| CIFAR-10 / ResNet-32 | 0.003 | 412--481 | 418 | 422 |
| WikiText-2 / Transformer | 0.003 | 784--852 | 790 | 794 |

These are exactly the manuscript's historical example windows and episode
markers, not newly selected fixed-reference events. The later fixed-reference
event dataset places pre-rise states at 689/415/787 and recoveries at 729/482/855;
therefore the original ResNet and Transformer crops end before that definition's
recovery. The gray shading and horizontal starting-loss references are retained
from the requested figure without relabeling them as the newer event definition.

## Added definitions and reduction

Row 5 is cos(g_t,m_t), using `gradient_current_moment_cosine`: the raw full-objective
gradient at parameter state t and the stored, uncorrected first moment after
incorporating that gradient. It is not cos(g_t,m_(t-1)) or an adjacent same-stream
cosine. At state t the moment/direction is used for outgoing update t -> t+1;
the recorded optimizer update number is t+1.

Row 6 contains both `moment_over_gradient_norm` = ||m_t||/||g_t|| and
`unbiased_moment_over_gradient_norm` = ||m_hat_t||/||g_t||, where
m_hat_t = m_t/(1-beta1^(t+1)), beta1=0.9. This row uses absolute same-state
norm ratios, not the ratio of the independently normalized curves in row 4.
Both ratios match exactly at recorded precision throughout these late windows;
the solid orange and dashed purple curves overlap. Both are retained explicitly.

All new measurements use the same whole-model parameter scope, FP32 stored
moments and FP64 norm/cosine reductions as Run215. The denominator is positive
at all displayed states; all 528 added values are finite. No smoothing,
subsampling, epsilon substitution, new normalization or event selection.
Loss and relative norms retain logarithmic axes; cross-cosine and norm ratios
use linear axes, with the ratio axis starting at zero.

## Caption

Zoomed full-batch Adam loss excursions on CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer. The original rows show training loss, projected gradient
change, adjacent raw-gradient and first-moment cosines, and their norms relative
to their respective crop-start values. Two added rows show current-gradient/
current-moment cosine and uncorrected/bias-corrected momentum-to-gradient norm
ratios. All points are unsmoothed. Gray shading, onset/peak markers and loss
references reproduce the manuscript's historical example definitions.

## Observation, uncertainty and limits

Current-gradient/current-moment cosine stays positive in these displayed windows,
despite adjacent gradients or moments reversing in parts of the same windows.
This illustrates distinct vector comparisons; it does not establish that
momentum causes or prevents a spike. Norm ratios and cosine signs alone do not
prove loss decrease, stability or generalization. The examples were selected
retrospectively from one initialization per model; repeated states are dependent
and no independent-seed uncertainty bands are available. Projected gradient
change under Adam is not an identified Hessian curvature or universal threshold.

## Verification and visual review

27 curves / 1,584 points pass source hashes, preservation checks, state/update
alignment, direct raw-record comparison for the added fields, independent norm
ratio identities, Matplotlib artist checks and CSV round-trip checks. Original
15 archived curves match exactly; the three original moment-cosine curves match
the full unsmoothed manifest and stored moments exactly. The 18 shaded panels
preserve the three historical intervals. One-page 7.16 x 7.8-inch PDF has embedded
fonts, vector curves and text inside page bounds. The full-size render was
inspected: six rows, panel labels, markers, axes and the four-entry legend are
legible with no clipping or overlap. The ratio curves intentionally coincide.
Scratch PNG remains ignored under `tmp/pdfs/analysis058/`.
