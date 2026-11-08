> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [005-technical-report-event-window.md](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/005-technical-report-event-window.md).

# Technical report and MLP event-window refit

## Conclusions first

The [five-page technical report](../report/instability-tracing-report.pdf) consolidates
motivation, the formal algorithm, matrix dimensions, probe decisions, results,
interpretation and costs. It uses **MLP outgoing updates 450-500 only**, freshly
refitting all SVD quantities. The original 448-800 evidence and Figure 03 remain
unchanged. [Event-window companion Figure 04](../figures/04-mlp-instability-trace-450-500.pdf)
uses the same two-page layout as the prior architecture reports.

In this shorter MLP window, all five learned representations and the raw-gradient
stream reverse at all 51 displayed updates. First-projection/output alignment
correlation is 0.9893; output top-one/top-ten energy is 53.02%/99.48%. The leading
coefficient changes sign 42 times over 50 adjacent pairs at every learned stage.
These locate repeated motion along fixed fitted patterns; they do not establish
a causal initiating layer or stationary local oscillation directions.

**A fixed SVD basis is not evidence that motion never rotates.** Each pattern is
fixed over its fitted window, but its relative coefficients can vary. Local
dominant axes/subspaces could rotate within that span. No local-window
principal-angle/rotation test has been performed. Cross-layer temporal overlap
does not answer that question.

## Source and reduction contract

The sources are the qualified seed-0 Adam continuation in Runs 240-242 and the
exact retained Grams in Analysis 062. The original full-objective 5,000-image
coverage, raw-output squared-error contract, FP32 captures and FP64 reductions
are preserved. No new training, sampling or cloud execution occurred.

`07_mlp_event_window.py` verifies the source hashes recorded by the existing MLP
adapter, then slices the original 354-by354 Grams. Original rows represent outgoing
displacements/raw gradients 447-800. Rows 449-500 (indices 2:54) supply a 52-by52
submatrix. Row 449 supplies the predecessor for rho 450; only rows 450-500 enter
the SVD, giving 51 fitted rows. Required representation states are 449-501.
Hidden/preactivation matrices have shape 51-by1,000,000; output has 51-by50,000;
raw gradients have 51-by656,810. Samples/features are never averaged, rows are
not normalized, and data are not centered. Gradients themselves are not differenced.

The basis is refitted once per stream, with its largest absolute temporal
coefficient oriented positive. Newly computed cosines/RMS agree with the
corresponding original per-update values; the new spectra, coefficients and
overlaps intentionally differ from the old window's quantities. Independent
singular-value reduction of each sliced Gram and full coefficient reconstruction
pass. See [numerical results and source hashes](../data/mlp_450_500_results.json) and
[stage table](../data/mlp_450_500_stages.csv).

| Representation | Negative cosines | First-mode energy | First-ten energy | Modes for 95% | Alignment correlation with output | Temporal O10 with output |
|---|---:|---:|---:|---:|---:|---:|
| First linear Z1 | 51/51 | 50.98% | 97.48% | 7 | 0.9893 | 0.7473 |
| First tanh H1 | 51/51 | 49.51% | 97.60% | 6 | 0.9928 | 0.7932 |
| Second linear Z2 | 51/51 | 50.60% | 98.80% | 5 | 0.9980 | 0.8994 |
| Second tanh H2 | 51/51 | 50.78% | 98.85% | 5 | 0.9986 | 0.8993 |
| Raw output F | 51/51 | 53.02% | 99.48% | 5 | 1.0000 | 1.0000 |
| Raw gradient g | 51/51 | 54.80% | 99.66% | 5 | 0.6420 | 0.9470 |

The requested interval contains the dominant loss peak 474 (loss 0.05367296) and
smaller oscillations. It is an event-scale interval, not a monotone single rise
and fall. MLP output displacement is largest on outgoing update 474; ResNet and
Transformer maxima remain 292 and 561, entering their loss peaks 293/562. These
different timings are stated explicitly in the report.

## Figure captions and generating scripts

**Companion Figure 04, page 1:** full-objective loss (equal to probe loss), raw
gradient/output alignment, all five representation-change cosine heatmap rows,
coordinate RMS divided by each row's own 450-500 maximum, top-one/top-ten SVD
energy and top-ten temporal overlap with output. **Page2:** per-stage cosine,
absolute coordinate RMS, leading fixed coefficient/sqrt(P). P includes all sample
and feature coordinates. Fixed inputs have zero motion and undefined cosine;
there is no artificial zero-cosine input row. Generator:
`02_figures.py --case mlp --mlp-event` after `07_mlp_event_window.py`.

**Report Figure 1:** three model columns with full/probe loss, raw-gradient/output
alignment, first/last/output landmark alignments, and their relative displacement
RMS. Model windows and output spaces are labeled; dotted lines identify the
maximum full-objective loss. First/last landmarks are linear1/hidden2,
conv1/final residual output, and embedding/final Transformer block.

**Report Figure 2:** all five MLP learned representations over 450-500, with
consecutive cosine, absolute coordinate RMS and leading coefficient/sqrt(P).
Its truncated cosine vertical range [-1.05,0.15] contains every plotted value;
the companion uses the full cosine range. No temporal smoothing is used.
Both report figures and the numerical LaTeX table come from
`08_report_assets.py`; their local vector sources are retained under `report/figures/`.
`09_build_report.py` compiles the LaTeX twice and checks five-page structure,
the rendered algorithm, reference resolution, overfull boxes and text bounds.

## Coverage, uncertainty and interpretation limits

This is a local reanalysis of selected single-seed trajectories, with no
confidence intervals or independent-probe replication. ResNet/Transformer
retain their original 31/49-row windows and subset probes; all numerical values
come directly from their verified results files. MLP's shorter 51-row fit makes
the event scale more comparable, but probe size, objective, output coordinates
and architecture remain different. Rank is bounded by the number of time rows.
Correlations are descriptive summaries of temporally dependent observations.

The MLP first-layer/output O10 falls from 0.9956 in the longer fit to 0.7473 in
this window, while its alignment correlation remains high. The shorter output
spectrum is also more concentrated (top-one 53.02%, versus 25.17%). These are
window-dependent summaries, not contradictory measurements or evidence of
rotation by themselves. Temporal similarity does not establish identical neuron
directions, uniform motion across hidden dimensions or a universal first-layer
cause. A causal intervention and a local-subspace rotation test remain separate
unperformed analyses.

Verification records: [MLP companion values](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/mlp_450_500_figure_verification.json),
[report assets](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/report_assets_verification.json),
[compiled PDF](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/technical_report_verification.json), and
[rendered-page review](../../../../../../analyses/063-2026-11-07-architecture-instability-tracing/report_visual_review.json).
The [workflow harness](../../../../../../research/INSTABILITY_TRACING.md) now explicitly
records fixed-basis versus rotating-motion semantics and the required window
refit. All work is local; no new run, hypothesis, worker or billable resource.
