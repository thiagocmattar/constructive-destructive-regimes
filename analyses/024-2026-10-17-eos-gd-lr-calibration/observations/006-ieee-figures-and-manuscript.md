# Observation006: IEEE figure revision and manuscript integration

## Question and sources

Present the existing Analysis024 evidence coherently after the manuscript's
parameter-displacement section, with publication-size artwork and an explicit
critical review of each of the six diagnostics. Scientific sources are the
unchanged nine conditions in `results.json`, their verified owning-run metrics
and spectra, the frozen cohort/inputs, and Observation005. No new experiment,
noise draw or training trajectory is introduced.

## Reduction, coverage and artifacts

`07_plot_single_rows.py` still generates the six isolated one-row, three-column
PDFs. It also generates two two-row, three-column composites:

- `figures/eos_gd_alignment_displacement_3k.pdf`: smoothed cosine above raw R_t.
- `figures/eos_gd_norms_projected_change_3k.pdf`: smoothed raw-gradient norms
  above smoothed eta*kappa_hat.

The 54 isolated Y arrays and their 160,659 points are unchanged. The composites
reuse 36 curves / 106,659 points; all eight PDFs contain 90 curve instances /
267,318 points. No new scientific reduction is introduced. The complete-window
50-observation trailing arithmetic mean is unchanged, including averaging
norms before log display and averaging pointwise q values.

Gradient diagnostics now use the manuscript's gradient index t: norms are
plotted at t=1..3000 and consecutive pairs at t=2..3000, with gradients
evaluated at theta[t-1]. Their previous gradient-state X coordinates move
by exactly one. The first complete norm/pair means are at t=50/51. Loss
remains at parameter states0..3000; R_t remains after updates1..3000.
This is an indexing alignment, not an extra update or gradient measurement.

All PDFs have 7.16-inch width, embedded Times New Roman/STIX fonts, solid LR
curves, external panel labels and no embedded footer. Method and coverage
information is supplied in the paper's methods and captions. Isolated figures
are 2.55 inches high; composites are 3.65 inches high. PDF scales and windows
are disclosed in `single_rows_results.json` and the accepted figure manifest.
`FIGURE_REVIEW.md` records the six individual assessments and remaining limits.

`08_build_manuscript_assets.py` copies four reviewed PDFs byte-for-byte into
`manuscript/draft/figures/analysis024/` and generates the nine-condition
`manuscript/draft/tables/analysis024_endpoints.tex` directly from `results.json`.
`manuscript_assets.json` pins the source and copied-output hashes.

## Manuscript claim-to-evidence map

| Location | Observation / source | Supported interpretation |
| --- | --- | --- |
| New Section IV-A | Frozen cohort/configs, input manifest and Analysis024 README | Three paired LR conditions per model, 3k full-GD updates, one frozen initialization; objective and spectral coverage explicit |
| Fig. 1 / Table I | Full objective loss states and verified endpoints | Loss decreases overall; largest tested LR has lowest final train loss within each case |
| Fig. 2(a)-(c), Appendix D / Fig. 8 | All 26,991 defined pairs, with declared smoothing only in the main view | Late reversal in seven conditions; raw transients differ from sustained regimes |
| Fig. 2(d)-(f) | All 27,000 complete-prefix R values | Relative cumulative displacement falls across the selected LR comparisons; no lag-one truncation |
| Fig. 3(a)-(c) | 26,559 smoothed norm points | Magnitude adds information about the realized motion; no norm-based stability criterion |
| Fig. 3(d)-(f), Table I | 26,550 smoothed q points, existing late means and residual-checked final spectra | Algebraic relation to cosine, finite-step directional interpretation and separation from maximum/subset curvature |

Each main caption gives the relevant smoothing, seed and coverage; no hidden
seed averaging or uncertainty band is introduced. Raw alignment is preserved
in Appendix D rather than serving as another purported independent test.
Only obvious Section III typos and the incorrect 'lag<=2' description of
nonadjacent updates are corrected; its formulas are unchanged.

## Uncertainty and caveats

One initialization per model, calibration-selected LRs and fixed finite horizons
support conditional descriptive comparisons. Three models over two datasets
do not isolate architecture from dataset/loss changes. The q/cosine relation
is algebraic, and smoothed crossing times are not raw first-event indices.
Only MLP spectral coverage matches the complete training objective. Late
means and final eigenvalues use different temporal reductions; disagreement
cannot identify its cause. R_t is path-relative and does not quantify learned
information or imply a negative all-pair sum. Full-GD evidence does not validate
moving-SGD noise claims, generalization, global stability, convergence,
prospective warning, a controller or negligible diagnostic overhead.

## Verification and outputs

Source/config/initial hashes, q and complete-prefix aggregation identities,
independent convolution and exact artist arrays pass. An independent closeout
check compares all 54 Y hashes to the pre-revision manifest, verifies the
one-index X change, inspects all PDF fonts/pages/bounds, and checks all copied
PDFs and table numbers. All eight Poppler chart renders and the compiled
manuscript pages are visually reviewed. Final build output is the existing
ignored `manuscript/artifacts/draft/bare_jrnl.pdf`, now 16 pages, with resolved
citations/references and no overfull boxes. Nonfatal template/legacy layout
warnings remain visible in the ignored build log.

The local verification record is `manuscript_verification.json`. Raw artifacts,
frozen inputs, the original overview and supplied manuscript assets are
preserved. No new resource or billing is involved. All changes are local.
