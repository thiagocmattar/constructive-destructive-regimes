# Observation005: complete nine-arm controlled Adam diagnostics

## Sources and completed comparison

Run215 completed all nine approved constant-LR arms, each from the frozen
Analysis024 initial tensors with fresh Adam moments. MLP rates are0.0001/0.0003/
0.001; ResNet-32 and Transformer rates are0.0003/0.001/0.003. Each has1,024
full-objective updates, for9,216 total. There is one initialization per case,
no schedule/intervention, clipping, decay, Hessian/HVP or seed sweep.

`runs/215-2026-11-02-controlled-full-batch-adam/results.json` is complete_verified
with no missing or incomplete condition. The input/config manifests, raw histories,
final model/Adam/RNG/diagnostic states and gradient audits at0/1/1024 remain local.
The full scientific verifier passed before deleting every production Pod and
passed again with --require-complete after retrieval. All9,225 raw boundary losses,
9,216 accepted-update records and81 full evaluation records are covered. Evaluation
uses complete10k-image test splits and all217,630 valid/245,550 test Transformer
targets at states0,128,...,1024. Losses are weighted by image or supervised target.

## Figure captions and reduction

`figures/adam_diagnostics_full.pdf`: complete controlled full-batch Adam diagnostics,
three columns for CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer.
Rows are training loss (log), raw-gradient eta*kappa_hat, consecutive rho for
raw gradients, stored momentum and adaptive Adam directions, raw-gradient and
stored-momentum norms (separate log rows), and cumulative R_t only for measured
Adam parameter updates. Each column has three increasing-LR colors. Loss and
R_t are unsmoothed; the six diagnostic rows use complete16-observation trailing
arithmetic means. The norm rows share vertical limits within each column.

`figures/adam_diagnostics_full_unsmoothed.pdf`: the identical nine-condition cohort
and row/column/color arrangement, with every finite recorded value in every row
plotted directly. No averaging, binning, thinning or downsampling is performed.
Its loss and Adam R_t arrays equal those in the other version exactly. Raw
projected-change limits expand to include all extrema. Dense oscillations can
overlap in a full-horizon view; zooming the vector PDF retains the raw points.

In both versions loss states cover0..1024. Vector rows correspond to applied
updates1..1024, consuming gradients from states0..1023; the separately measured
unapplied terminal gradient is excluded. Undefined first adjacent measurements
are omitted without filling. The raw-gradient projected expression is computed
pointwise before any smoothing. Its dashed reference at2 comes from the GD
visual pattern and is not an identified Adam stability threshold. R_t uses net
measured displacement divided by cumulative measured update length; it is never
reset or smoothed. Observation003 gives the metric definitions in full.

## Verification, coverage and limits

`10_plot_completed_preview.py` generates the first version from the verified
results, checking source bytes/hashes, indices, pointwise identities, window
means and artist data. `11_plot_unsmoothed_alternative.py` uses that exact cohort's
raw arrays and rechecks the immutable histories. The source identities and every
plotted value are in full_diagnostics_manifest.json/full_unsmoothed_manifest.json.
`12_verify_full_figures.py` independently checks both reductions and PDFs.

Both figures have72 curves across24 panels. The first has72,891 plotted values;
the fully unsmoothed version has73,701. Scalar math.fsum window means agree to
1.78e-15 absolute error. All raw values and step indices are exact; all losses
and actual-update R_t values agree between versions. Both one-page PDFs pass
text-boundary/footer checks and final visual review, without clipped labels or
overlapping text. Renders remain ignored scratch. The earlier paired six-arm
previews are retained separately.

One trajectory per case/LR gives no across-initialization uncertainty or confidence
bands. MLP uses the same owned5070Ti laptop across its rates. ResNet uses aPRO4000,
aPRO6000 partition and anRTX4090 respectively; Transformer uses RTX4090/PRO4500SE/
RTX4090. Every cloud worker passed initial tensor/gradient qualification, but
hardware variation and subsequent FP32 drift limit exclusive LR attribution.
The projected raw-gradient expression under Adam does not measure Hessian
curvature or identify a stability boundary. Alignment or displacement retention
alone does not establish loss reduction, convergence or generalization. Final
held-out losses are retained separately from the plotted training objective.

## Execution closeout

All10 main-stage task Pods/disks (six production and four pre-production retries)
were deleted after verified retrieval or verified absence of scientific input.
A fresh account read at2026-11-02 16:54UTC confirms zero Analysis036 Pods; the
five calibration Pods were also already deleted. Retained task cloud storage is
0GB. Main compute/storage estimate isUSD11.62365; calibration isUSD2.26282;
combinedUSD13.88647, provisional pending final billing. No guard triggered.
See Run215 completion.json, production-closeout.json and retrieval receipts.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/10_plot_completed_preview.py --output-stem adam_diagnostics_full --manifest-name full_diagnostics_manifest.json
    python analyses/036-2026-11-02-controlled-adam-diagnostics/11_plot_unsmoothed_alternative.py --source-manifest full_diagnostics_manifest.json --output-stem adam_diagnostics_full_unsmoothed --manifest-name full_unsmoothed_manifest.json
    python analyses/036-2026-11-02-controlled-adam-diagnostics/12_verify_full_figures.py
