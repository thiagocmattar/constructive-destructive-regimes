# Observation002: Adam at largest-rate GD states, consecutive lags 1–256

Requested view: reproduce Analysis033's `figures/largest_lr_lag256.pdf` with
the Adam continuations, retaining the largest source GD rate per architecture
and one existing full-production Adam rate. The user explicitly chose MLP
3e-4 and ResNet-32/Transformer 1e-3 after clarification: MLP 1e-3 was tested
only in 64-update calibration, failed eligibility, and has no 256-lag curve.
These are the calibrated rates and the largest rates available in full
production, not the largest candidate tested for every architecture.

Sources: Analysis034 `results.json`, independently exported `lag_profiles.csv`,
and `calibration_decision.json`. MLP uses parent Run122 (GD LR 0.04), ResNet-32
uses Run120 (0.1), and Transformer uses Run126 (0.2), each at retained anchors
500, 2000 and 3000. All nine selected starts are directly saved and primary;
the two unqualified Run062 reconstructions are outside this subset. Production
horizons cover at least 256 updates for every selected profile. The reference
layout and generator are Analysis033's `12_plot_largest_lr_lag256.py` and PDF.
Input, reference, generator, plotted-value, PDF and render SHA-256 identities
are recorded in `largest_lr_lag256_review.json`.

Reduction: select the nine profiles and take their exact `anchor_cosine` at every
integer lag 1 through 256, for 2304 points. Connect each panel's points in lag
order as one curve. No parity split, averaging, smoothing, projection, envelope
or downsampling. Use the reference's 3×3 layout, 12×8.8-inch nominal size,
blue line width 0.7, zero reference, linear x=1..256 and y=-1.05..1.05, ticks
and typography. Change optimizer labels to Adam and show both numerical rates.
Every plotted array is checked against the JSON and CSV, including the source
rates and state = anchor + lag. All cosines are finite and within [-1,1].
The exported PDF is one vector page, has no raster images and has all text
within its page; its actual PDF render passes visual inspection.

Caption, `figures/largest_lr_lag256.pdf`: Whole-model raw-gradient cosine
cos(g₀,gₖ) along cold-Adam continuations from the largest-rate GD histories.
Columns show CIFAR-10 MLP (Adam LR 3e-4, source GD LR 0.04), CIFAR-10 ResNet-32
(1e-3, 0.1), and WikiText-2 Transformer (1e-3, 0.2). Rows show retained states
500, 2000 and 3000. Here g₀ is the raw gradient at the retained anchor; k counts
Adam updates after that state. All lags 1–256 are connected in order, with a
horizontal zero reference. The curves measure orientation relative to the
initial gradient of each fork; they are not adjacent-gradient cosines or Adam
update-direction cosines.

Result and limits: this view displays the initial anchored oscillations and
their later smaller excursions about zero over the requested window. It does
not establish permanent or asymptotic damping. Full-objective gradients,
parameter scope, FP32 gradient accumulation and exact FP64 cosine reductions
are unchanged from Observation001; no gradient compression is used. There is
one seed-0 initialization per architecture and nine dependent checkpoint forks,
with one anchored pair per lag and no uncertainty bands. Adam moments reset
to zero; the optimizer and its numerical LR both change from GD, and actual
step norms are not matched. The view alone does not establish optimizer
superiority, stability or generalization.

Generator: `13_plot_largest_lr_lag256.py`. Publication output:
`figures/largest_lr_lag256.pdf`. Actual-PDF preview:
ignored `scratch/largest_lr_lag256.png`. This is a local replot of existing
measurements with no new training, remote resources or cloud charge.
