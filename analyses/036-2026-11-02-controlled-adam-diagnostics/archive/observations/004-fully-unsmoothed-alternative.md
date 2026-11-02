# Observation004: fully unsmoothed alternative to the six-run preview

## Sources and coverage

The user requested an alternative with no smoothing in any row. It uses exactly
the six completed conditions frozen in `completed_preview_manifest.json`, rather
than advancing to newer completed runs during the comparison: MLP LR0.0001,
0.0003 and0.001; ResNet LR0.0003 and0.003; Transformer LR0.0003. Each has1,024
updates. Their immutable histories are checked by SHA256 again. The baseline
manifest contains the original raw arrays and pins its historical cohort.

`11_plot_unsmoothed_alternative.py` generates the PDF. Sources, complete plotted
arrays, output hash and axis limits are in `completed_unsmoothed_manifest.json`;
independent source-equality and PDF checks are in `completed_unsmoothed_verification.json`.

## Caption and reduction

Fully unsmoothed Adam diagnostics for the same six completed Run215 arms as the
partially smoothed preview. Columns and eight rows retain the approved layout,
notation, LR colors, log scales and separate gradient/momentum norm rows. All
finite observations are plotted directly, including all1,025 training losses
and all1,024 measured Adam-update R_t prefixes per arm. The initial undefined
adjacent cosine/projected-change observations are omitted without filling them.
There is no averaging, binning, thinning or downsampling. R_t remains the recorded
cumulative statistic; only the actual measured Adam update stream is used.

The horizontal clock and metric definitions in Observation003 apply unchanged.
Training loss and R_t are identical to that preview. Its other six rows have
their16-observation means replaced with the original values. The projected-change
axis expands to include every raw extremum; the dashed reference at2 remains a
numerical GD reference, without an Adam stability or Hessian interpretation.
Gradient/momentum norm rows share a vertical range within each column.

## Verification and limits

All48 curves /49,134 values and their step indices equal the finite source
observations exactly. Artist arrays are checked directly. Loss and Adam R_t
also equal the corresponding partially smoothed preview arrays exactly. Every
raw projected-change value lies inside the displayed limits. The single-page PDF
passes text-boundary/footer checks and final visual review, with no clipped labels
or overlapping text. Its PNG render remains ignored scratch.

Coverage is3/3 MLP,2/3 ResNet and1/3 Transformer, one initialization per case and
no uncertainty bands. The same hardware qualifications and missing-rate caveats
as Observation003 apply. Dense raw oscillations can overlap at this full-horizon
scale. No new training, cloud spending or scientific interpretation is introduced.
The partially smoothed preview remains available as a separate artifact.

Reproduce:

    python analyses/036-2026-11-02-controlled-adam-diagnostics/11_plot_unsmoothed_alternative.py

Output: `figures/adam_diagnostics_completed_unsmoothed.pdf`.
