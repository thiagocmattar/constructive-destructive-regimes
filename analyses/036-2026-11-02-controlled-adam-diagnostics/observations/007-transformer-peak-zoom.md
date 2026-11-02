# Observation007: unsmoothed Transformer Adam loss-peak zoom

## Sources and window selection

The user requested all eight statistics from the complete unsmoothed Adam figure
for WikiText-2/Transformer at LR 0.003, around the visible loss peak near step 768,
plus step-by-step loss differences. This reuses Run215's completed, verified
full-objective trajectory without new training, measurement or cloud spending.

The prominent peak is at parameter state 794, with training loss
2.1455662476275608. It is selected as the largest recorded loss in states
704 through 832, a +/-64-state neighborhood around the approximate requested
anchor 768. Small local maxima at 769 and 774 are distinct from this prominent
spike. The search rule and every endpoint value are recorded in
`transformer_peak_zoom_manifest.json`.

The displayed window starts ten steps before the selected peak, at state 784.
The phrase "value before the peak" is operationalized as the loss at that first
displayed state: L_784 = 0.6522930247178894. Search strictly after the peak for
the first loss strictly below this reference. This occurs at state 852, where
L_852 = 0.6518715207097087; the preceding loss is
L_851 = 0.6541558096503138. Every state from 795 through 851 remains at or above
the reference. The crop therefore contains 69 consecutive states, 784 through
852 inclusive. Earlier values below the reference during the pre-peak interval
do not qualify as a post-peak return.

## Caption and reduction

`figures/adam_transformer_peak_zoom.pdf`: WikiText-2/Transformer trained with
constant Adam LR 0.003, zoomed to steps 784-852. The nine vertically aligned
panels show training loss, step-by-step loss difference L_t-L_(t-1), raw-gradient
eta*kappa_hat, consecutive raw-gradient rho_t, momentum rho_t, adaptive Adam
rho_t, raw-gradient norm, stored-momentum norm and cumulative actual-Adam-update
R_t. Every original curve is unsmoothed and drawn with individual point markers.
Loss differences are bars: positive orange, negative blue. The dashed vertical
line marks the loss peak at 794; the dotted horizontal loss reference is L_784.

All eight original statistics are exact slices of
`full_unsmoothed_manifest.json`, corresponding to the verified
`figures/adam_diagnostics_full_unsmoothed.pdf`. No averaging, interpolation,
thinning, binning or re-estimation is performed. Loss and both norm rows retain
logarithmic Y axes. Cosines retain the full [-1.08,1.08] display range. Other
limits are fitted locally, including R_t: its displayed axis does not start
at zero. The two norm rows use their own local limits.

The original clock is preserved: loss L_t is evaluated at parameter state t;
vector rows at t belong to accepted update t and use the raw gradient from
state t-1. Momentum is the actual stored first moment, and the Adam cosine is
for the bias-corrected adaptive direction. The projected expression is the
original pointwise gradient-change diagnostic, without an Adam curvature claim.
R_t uses measured actual parameter increments and retains its cumulative prefix
from initialization; it is not reset at state 784. The separately measured
unapplied terminal gradient is outside this crop.

There are 68 loss differences, at t=785 through 852, each subtracting the two
adjacent losses inside the displayed window. No loss outside the crop is used
for this added panel. Each difference is the direct subtraction of recorded
full-training-objective losses, without relative normalization or clipping.

## Verification, coverage and limits

`14_plot_transformer_peak_zoom.py` produces the PDF and manifest. It verifies
all used source SHA-256 hashes, the prior full-figure lineage, the selected arm's
completed status, loss-state/update indices, the peak and first post-peak return,
all eight exact raw slices, Matplotlib artist values and every loss difference.
The differences also telescope to L_852-L_784 within 2e-15 absolute tolerance.

The figure has nine panels, 552 original observations (69 per statistic) and
68 loss differences, for 620 displayed values. All original values and indices
are exact. The single-page PDF passes text-boundary checks and Poppler visual
review: individual points, bars, labels, reference lines and footers are legible
without overlap or clipping. The render remains ignored scratch.

This is a retrospective view of one deliberately selected spike from one
trajectory and initialization. It supplies no across-initialization uncertainty,
frequency estimate for spikes or causal explanation of the loss response.
Projected raw-gradient change under Adam does not identify Hessian curvature or
a stability boundary; alignment and retention alone do not establish loss
decrease, convergence or generalization. Earlier figures and raw evidence are
preserved. No scientific input or executed training budget changes.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/14_plot_transformer_peak_zoom.py
    pdftoppm -singlefile -scale-to 2000 -png analyses/036-2026-11-02-controlled-adam-diagnostics/figures/adam_transformer_peak_zoom.pdf analyses/036-2026-11-02-controlled-adam-diagnostics/scratch/adam_transformer_peak_zoom
    python analyses/036-2026-11-02-controlled-adam-diagnostics/14_plot_transformer_peak_zoom.py --accept-reviewed
