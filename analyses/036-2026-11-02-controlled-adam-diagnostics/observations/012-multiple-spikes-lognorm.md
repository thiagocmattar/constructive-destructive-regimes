# Multiple-spike figure with logarithmic relative norms

Date: 2026-11-05. Author-requested presentation alternative to
[the linear-norm figure](011-multiple-spikes.md).

Output: [log-norm PDF](../figures/adam_manuscript_spikes_multiple_lognorm.pdf).
Generator: [13_plot_multiple_spikes.py](../13_plot_multiple_spikes.py).
Exact arrays and provenance: [manifest](../multiple_spikes_lognorm_manifest.json).

```powershell
python analyses/036-2026-11-02-controlled-adam-diagnostics/13_plot_multiple_spikes.py --norm-scale log
```

Sources, normalization, full-objective reductions, clock, event definition and
scope follow Observation011 and the same SHA-256-pinned Run215 histories.
The twelve unsmoothed green curves / 3,608 values are unchanged. Only the
relative raw-gradient norm row now uses logarithmic y-axes; every value is
strictly positive, so no clipping, offset or omitted state is needed. Norms
remain divided by their value at the first displayed state. Loss remains
logarithmic; projected change and cosine remain linear. No momentum curves
or gradient/momentum legend are included.

Coverage remains MLP 465-744, ResNet-32 215-495 and Transformer 525-865,
with all 19 complete >10% excursions and identical gray pre-rise-to-peak
bands. The linear-norm PDF is retained byte-for-byte; its manifest is
refreshed to reference the generator's new optional scale argument.

Caption: Repeated full-batch Adam loss excursions at each model's highest
tested rate. Rows show loss, projected raw-gradient change, consecutive
raw-gradient cosine and relative raw-gradient norm. Loss and relative norms
use logarithmic y-axes. Each norm is divided by its first displayed value.
All curves are green and unsmoothed. Gray bands mark pre-rise to peak for
complete excursions exceeding 10%. At parameter state s, diagnostics refer
to g_(s+1). One initialization per model.

Verification: source hashes, archived source arrays, projected-change identity,
event boundaries, positive norm values, exact plotted arrays, embedded fonts
and PDF text bounds pass. All twelve curves, windows and event bands match
the linear-norm manifest exactly apart from the three norm-scale fields.
The PDF was rendered with Poppler at 2,200-pixel width and visually reviewed
for tick legibility, label spacing and clipping.

Uncertainty and caveats are unchanged from Observation011: these are selected
single-initialization descriptive windows, without independent-seed intervals,
new curvature evidence, predictive validation or causal inference. No new
training, run, hypothesis, cloud work or manuscript edit.
