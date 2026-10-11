# Observation006: loss, smoothed raw norms and momentum by learning rate

Date: 2026-10-10. User-requested presentation of existing Analysis009 results,
with training loss first and raw-gradient/momentum norms in separate LR panels.
The current display revision adds a trailing 10-update raw-norm average, faint
original raw traces and a fill between the average and unsmoothed momentum.
No training, scientific-input change or new hypothesis is introduced.

Sources: [comparison data](../comparison_data.json), SHA-256
`a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739`,
containing the verified scalar histories for Runs021-027. The
[plotted data and provenance](../loss_and_norms_by_learning_rate_data.json)
pin source, generator, value/palette helpers and PDF hashes, and retain all
plotted values and per-update supervised-token counts. Raw measurements and
other PDFs are preserved; the original unsmoothed presentation is retained in
Git commit `bcb6768`. Generator: [14_plot_loss_norms_by_lr.py](../../14_plot_loss_norms_by_lr.py),
using the existing Analysis009 palette and Analysis008 scalar extraction.

Caption: **Unsmoothed training loss, smoothed raw-gradient norms and stored
momentum norms across seven
warmup-only learning rates in Pythia-14M.** The
[single-page PDF](../figures/loss_and_norms_by_learning_rate.pdf) uses a 4x2 grid.
Panel (a) overlays the recorded training loss at every accepted update for all
seven LRs, using the established reversed-plasma colors; dashed loss curves
identify Spark runs, solid curves RTX4090 runs. Panels (b)-(h) contain one LR
each, in increasing order: 1e-4, 0.000215443469, 0.000464158883, 1e-3,
0.002154434690, 0.004641588834, 1e-2. Each shows original raw-gradient norms in
faint blue (opacity 0.20), their trailing 10-update arithmetic average in solid
blue above them, and unsmoothed stored-momentum norms in dashed orange. A blue
fill (opacity 0.14) spans the raw-norm average and momentum; it is a visual gap,
not an uncertainty band. Norm panels share logarithmic
Y limits; training-loss Y and update X are linear. Panel titles identify LR
and hardware; the top legend applies to loss and the local legends to norms.

For raw norm r_t = ||g_t||, the displayed average is
sum(r_j, j=max(1,t-9),...,t) / min(t,10). Updates 1-9 use the available prefix;
every update has an averaged value. This averages vector norms in their original
units before logarithmic display, not gradient vectors or log norms. The window
matches the existing norm presentation in Observation003. Training loss and
momentum remain unsmoothed. There is no normalization, thinning or added positive
floor. Training loss is the recorded mean
cross-entropy over supervised tokens within each update, in nats. Raw g is the
whole-model accumulated/unscaled gradient before clipping. Momentum m is the
actual stored uncorrected Adam first moment after the optimizer update, using
clipped gradients and preceding bias correction/preconditioning. These are
distinct vector streams, not the realized AdamW parameter displacement.

Coverage includes **all 712 accepted updates**, including the zero-LR first
update. Circles identify the partial final update712, with311 sequences and
636,033 supervised targets versus1,024 sequences and2,096,128 targets for
full-sized updates. It contains fewer supervised tokens and is labeled separately.
This differs from preceding main figures, which display updates1-711 while
retaining712 in their data. The figure contains4,984 loss values and9,968 norm
values plus4,984 averaged raw-norm values. There are no zero or undefined norms
to mask on logarithmic Y. The separate final/minimum metrics continue to use
the original unsmoothed raw norms, not this displayed average. Their provenance
receipts are refreshed to the revised norm-panel data; their values and PDFs
are unchanged.

[Verification](../loss_and_norms_by_learning_rate_verification.json) checks
all21 measurement series directly against source fields and the rendered line
data, each average against independently summed windows, and the fill vertices
against the average and momentum arrays. It also checks full update coverage,
seven common logarithmic norm axes, source/output hashes,
PDF dimensions, text/page bounds and the rendered layout. The one-page16x18
inch figure was inspected for complete labels, LR/stream placement, shared
scales, legends, faint traces, fill placement and absence of clipping or overlap.
Both PyMuPDF and Poppler renders were inspected after regeneration.

The plot exposes different raw and momentum magnitudes along each trajectory;
the largest LR has increasing late raw norms while stored momentum remains
smaller. This is descriptive. Clipping at1.0 and EMA accumulation both affect
the norm comparison, so the gap does not isolate estimation noise or a causal
momentum benefit. One seed1234, frozen initialization/order and one full pass;
different hardware across old/new groups and LR-scaled decay limit cross-LR
causal interpretation. No uncertainty band, stability threshold, convergence
guarantee or generalization conclusion is introduced.
