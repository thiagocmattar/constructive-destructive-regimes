# Observation004: temporal interpretation and reviewer-facing revision

Date: 2026-10-30. The user requested another draft pass after questioning the
frozen comparison, negative alignment, the full-batch endpoint and loss/displacement
trends. This is postprocessing of Run186, with no new training or cloud execution.
It supersedes Observation003's manuscript presentation, not its measurements.

## Scientific assessment

The experiment varies batch size at one learning rate, 0.02. It cannot establish
learning-rate dependence. The three checkpoints are starting states for separate
200-update continuations, not labels for shared dynamical regimes.

Frozen independent gradient draws cannot test negative expected cosine. If the
draws are nonzero, normalize each draw to unit length. Independence makes the
expected inner product of two normalized draws equal the inner product of their
means; identical distributions make this a squared norm, which is nonnegative.
This is exact, whereas Eq. (19) approximates that expectation using full-gradient
and covariance moments. The prediction does not reuse the measured cosine, but
agreement is an approximation calibration, not new evidence about destructive
SGD. Exact moment identities serve as implementation checks. The maximum cosine
error 0.0075 and 21/36 conditional intervals excluding the prediction remain explicit.

The blue and orange streams in the moving figure measure different gradients
along the same SGD path. Orange is a full-gradient measurement, not a separate
full-GD trajectory. Only batch 5,000 supplies that control. Their discrepancy is
not an additive decomposition of sampling noise and trajectory effects.

At the 1,000-update state, batch 2,048 gives sampled/full mean cosine
-0.617/-0.630, compared with +0.839 under full GD. Full-gradient cosine becomes
negative within 3-5 continuation updates in all 24 stochastic repetitions;
its mean over updates 2-20 is -0.355. Full-GD first-negative reference indices
are 1,149/1,227/1,224. Only seed 0 reaches this event inside the 1,000-1,200
window: full-GD continuation means are 0.516808/0.999995/0.999981. This explains
the positive endpoint as a window-dependent summary of changed dynamics. It
does not identify a discontinuity, critical batch size or curvature mechanism.

At batch 32, full-gradient alignment also evolves during the continuation:
the all-window mean is approximately zero, despite negative early and positive
late means. A near-zero temporal average must not be read as stationary
orthogonality. The temporal figure retains this contrast without a new claim
about random-walk behavior.

Across increasing batch sizes, mean R rises monotonically at starting states
0 and 1,000, but stays approximately 0.03 at state 2,000. Mean loss improvement
is monotonic in the first two comparisons; batch 2,048 outperforms full batch
at the late state in every initialization-specific mean. These are equal-update,
unequal-data-exposure comparisons on the complete fixed 5,000-image objective.

## Manuscript changes

- Section III-B now states the domain and nonnegative expectation of frozen
  independent draws and distinguishes Eq. (19) from an exact identity. No
  equation, symbol definition or equation number changes.
- Section V explicitly specifies one LR and the fixed 5,000-image subset.
- Frozen calibration occupies two paragraphs, with its former figure retained
  in the research record rather than the compiled manuscript.
- The unchanged outcome figure is now Fig. 4. Its caption identifies full GD
  only at the full-batch endpoint.
- New Fig. 5 resolves selected 1,000-update continuations in time, followed by
  an interpretation of the observation window, destructive alignment and
  separate displacement/loss outcomes. No HVP material or appendix is restored.

## New figure contract

Artifact: [sgd_alignment_from_state1000.pdf](../figures/paper/sgd_alignment_from_state1000.pdf).
Generator: [11_temporal_paper_figure.py](../11_temporal_paper_figure.py).
Sources: Run186 attempt-01 production `metrics.jsonl` files for state 1,000,
batches 32/2,048/5,000, all three initializations and all repetitions. The three
reference histories supply the reported first-negative indices. All 54 source
files are checked against the previously verified immutable inventory SHA-256.

Selection: smallest batch, largest measured stochastic batch and full-batch
control. All five batches remain in Fig. 4; selection resolves its particular
endpoint contrast rather than replacing the complete comparison.

Reduction and coverage: 51 continuations of 200 updates, 199 defined pairs each.
Average the eight stochastic repetitions within each initialization (one for
full GD), then compute complete trailing 20-pair means. The first plotted value
is at continuation update 21, and the last at 200: 180 values per curve.
Average the three initialization curves equally; shade their minimum/maximum.
The six curves contain 1,080 plotted means and 3,240 initialization-level values.
No smoothing or truncation is applied to the numerical event or full-window claims.

Caption: alignment during continuations from the 1,000-update state, using the
smallest batch, largest stochastic batch and full-batch control. Curves show
trailing 20-pair means; shading is the range of initialization means. Full
gradients are evaluated along each SGD path and coincide with sampled gradients
under full GD. The X axis counts updates after the starting checkpoint.

Uncertainty: ranges describe three initialization means, not confidence bands.
Repeated sampling histories are nested within initialization. Smoothing obscures
individual first-negative events; those are separately checked from unsmoothed
histories. Averaging can mask initialization-specific transitions, illustrated
by the broad late full-GD range. No causal curvature or general stability claim
is inferred from the plot. The conditional first-negative observations do not
by themselves establish a sustained regime.

## Verification and visual review

[12_verify_temporal_revision.py](../12_verify_temporal_revision.py) independently
reconstructs the plot by direct window averaging rather than convolution. It
checks all 3,240 initialization values, 1,080 means and 2,160 range endpoints,
the existing outcome figure's 180 initialization values/60 means, 16 rounded
outcome statements, frozen error/range statements and all temporal events.
It checks copied asset hashes, embedded fonts, unchanged equations and the
unchanged outcome PDF. New notation and diagnostic tables remain absent.

The new 7.16-by-2.55-inch vector PDF uses the existing blue/orange palette,
solid curves, 9-point Times/STIX labels, external panel letters and no footnotes.
It is reviewed individually at 150 dpi and in the compiled draft at 100 dpi.
Axes cover all plotted ranges; full-GD overlap is explicitly explained. Both
main figures fit on page 10 before Section VI on page 11. Section V starts
on page 9. Section III-B is reviewed on page 5, its transition on page 6,
and remaining pages checked for reflow. The draft remains 20 pages with no
overfull boxes or unresolved references.

[temporal_figure_manifest.json](../temporal_figure_manifest.json),
[temporal_revision_verification.json](../temporal_revision_verification.json)
and [temporal_revision_review.json](../temporal_revision_review.json) retain
sources, arrays, checks and artifact hashes. Ignored snapshots, build logs and
renders are under `manuscript/artifacts/draft/noise-temporal-revision/`.
This review addresses the sampling argument and its connection to Section III;
it does not revalidate all legacy optimizer results or certify submission readiness.
