> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-controlled-minibatch-noise.md](../../../../../analyses/027-2026-10-30-minibatch-noise/observations/001-controlled-minibatch-noise.md).

# Observation001: controlled sampling noise and actual SGD motion

Date: 2026-10-30. H023; Run186 attempt-01. Status: completed, independently
audited and visually reviewed. [Report](27-REPORT.md) gives the interpretation;
this observation records the figure evidence contracts.

## Sources and reproducibility

Scientific inputs are frozen in [Run186](../../../../../runs/186-2026-10-30-mlp-minibatch-noise/README.md):
`config.json`, `input_manifest.json`, and `01_experiment.py`. Config SHA-256 is
`106bb149e32f7c3ea407c32868e2e45c0dd397447bd5cf83a0496c20a38f7d9b`;
executed script SHA-256 is
`ba5c9c132f58993a79edb5172a49094008c81a9dc31baa37456f389d734654ea`.
Production inventory SHA-256 is
`22367dfbfb0f2d6bcd0dc20ef3bd9296f657af5d8a4d25c2b3df216530113553`.

Raw production sources reside under
`runs/186-2026-10-30-mlp-minibatch-noise/artifacts/attempt-01/production/`:
reference histories/checkpoints, frozen sequence metrics and indices, moving
`metrics.jsonl`, indices, probe/final states, covariance/HVP measurements and
recovery state. They are local ignored evidence, not distributed in Git.
[02_verify.py](../../../../../analyses/027-2026-10-30-minibatch-noise/02_verify.py) audits coverage, indices, pairing, recovery/source
integrity, reference qualification, parameter motion and exact accounting, then
creates [results.json](../../../../../analyses/027-2026-10-30-minibatch-noise/results.json) and [verification.json](../../../../../analyses/027-2026-10-30-minibatch-noise/verification.json).
[03_analyze.py](../../../../../analyses/027-2026-10-30-minibatch-noise/03_analyze.py) generates all five figures,
[reduction.json](../../../../../analyses/027-2026-10-30-minibatch-noise/reduction.json), HVP residuals and the array/figure manifest.
[06_report.py](../../../../../analyses/027-2026-10-30-minibatch-noise/06_report.py) generates the scientific summary and two tables.
Bootstrap seed is 20261030, with 4,000 resamples.

Coverage: three initializations, saved GD states 0/1,000/2,000, five batch sizes,
32 frozen sequences of 128 draws per cell; eight moving sampling repetitions
per stochastic cell and one full-GD control. Totals: 184,320 frozen observations,
297 forks, 59,400 continuation updates, 59,103 defined adjacent moving pairs,
1,188 state-matched HVP pairs, and 6,000 separate reference updates. Complete
gradients, covariance and losses cover 5,000/5,000 training examples. No test
loss is claimed. All metric denominators are nonzero in this evidence.

## Figure contracts

### Frozen calibration

Artifact: [frozen_calibration.pdf](../support/frozen_calibration.pdf).
Sources: frozen metrics, full gradients and population covariance at the nine
reference states. Reduction: sequence means of noise squared norm, unnormalized
noise projection and adjacent cosine; average 32 independent sequences within
each initialization/state/batch cell. Both moment axes divide by the same
`S + V_b`; predictions are computed per cell before aggregation. The third
panel compares with `S/(S+V_b)`.

Caption: Frozen-state sampling calibration across three initializations and
three GD checkpoints. Colors denote batch size and markers denote checkpoint.
Each point is one saved-state/batch condition; bars are conditional 95% sequence
bootstrap intervals. Gray diagonals indicate agreement. Full-batch controls are
deterministic. Parameters remain fixed.

Observation: maximum relative discrepancies are 0.50% for variance and 0.74%
for noise projection; maximum absolute cosine approximation error is 0.00745.
The cosine target falls outside 21/36 stochastic intervals. Caveat: exact
moment identities and approximate cosine are different claims; this experiment
does not measure actual displacement. Repeated full controls are not replication.

### Moving geometry

Artifact: [sgd_geometry.pdf](../../../../../analyses/027-2026-10-30-minibatch-noise/figures/sgd_geometry.pdf).
Sources: every moving metric row, with sampled and complete gradients evaluated
at identical SGD states. Reduction: complete trailing 20-pair arithmetic means
for both cosine streams, starting at gradient index 21; no temporal smoothing
for `R_K`. Average repetitions within each initialization, then weight the three
initializations equally. Bands are the range of initialization means.

Caption: SGD continuations from shared GD states. Columns identify checkpoint;
rows show sampled cosine, complete-gradient cosine along the same SGD paths,
and normalized displacement from the fork. Colors identify batch size; black
is the separate full-GD continuation. Shading is a descriptive three-seed range,
not a confidence interval.

Observation: at the initial fork, batch 32 sampled/full-path cosine means are
−0.141/+0.353, versus +0.994 for full GD. At the later fork, weakening negative
sampled alignment does not necessarily improve normalized accumulation or loss.
Caveats: trajectory changes accompany sampling changes; cosines are not additive;
common checkpoint indices are not common dynamical regimes. `R_K` is path-relative.

### Raw moving geometry

Artifact: [sgd_geometry_raw.pdf](../../../../../analyses/027-2026-10-30-minibatch-noise/figures/sgd_geometry_raw.pdf).
Sources and initialization/repetition reduction match the preceding figure,
but every defined cosine pair remains temporally unsmoothed.

Caption: Unsmoothed counterpart of the moving-geometry comparison. Curves and
bands use the same repetition means and descriptive initialization ranges.

Observation: the main separation of sampled and full-path alignment is present
without temporal smoothing. Caveat: repetition averaging still suppresses
individual sampling-path variability; retained conditional bootstrap summaries
and raw histories supply that additional information.

### Full-objective loss

Artifact: [sgd_full_loss.pdf](../../../../../analyses/027-2026-10-30-minibatch-noise/figures/sgd_full_loss.pdf).
Sources: all full-objective state losses and the final-state measurement,
201 values per continuation. Reduction: no temporal smoothing; same equal-weight
initialization means/ranges as moving geometry. Logarithmic Y axes use local ranges.

Caption: Complete-objective training loss for the matched continuations. Columns
denote the fork checkpoint. Colors denote batch size; lines and shading show
the mean and range of three initialization means after repetition averaging.
All 5,000 examples contribute, including the final state.

Observation: at fork 2,000, batch 32 increases mean loss in all three blocks,
whereas full GD decreases it; batch 2,048 ends below full GD in all blocks.
Caveats: equal updates are unequal data exposure; no universal batch ordering,
generalization or efficiency conclusion follows. Compare numeric changes rather
than slopes across panels with different scales.

### Proxy and directional HVP validation

Artifact: [sgd_proxy_validation.pdf](../support/sgd_proxy_validation.pdf).
Sources: raw `q`, realized `epsilon` from full gradients, and HVP probes at states
preceding gradients 1/51/101/151; matching pairs are 2/52/102/152. Reduction:
20-pair curves use the moving-geometry rule. Filled corrected dots and open HVP
rings average **raw matching pairs** across repetitions and initializations.
The generated error table pools individual unsmoothed pairs by batch size.

Caption: Observed gradient-change proxy above and noise-corrected projection
below. Smoothed curves provide temporal context; raw paired markers compare the
corrected projection with independently evaluated complete-objective directional
HVPs. Quantitative errors use individual raw pairs.

Observation: at batch 32, median absolute HVP discrepancy falls from 0.7467 to
0.0030 after noise correction (288 pairs); corrected 90th percentile is 0.0143.
All stochastic batches improve. Caveats: the accounting identity is algebraic;
HVP residuals assess the finite-step approximation. The sampled direction and
visited state are retained. Full gradients make this an offline reference;
neither maximum eigenvalues nor an online corrected estimator are claimed.

## Validation and limits

The largest noise-accounting identity residual is 8.89e-16. Maximum relative
actual-versus-gradient-sum displacement discrepancy is 2.46e-4; the largest
endpoint difference in normalized ratios is 3.01e-8. These record FP32 update
roundoff, rather than asserting exact stored-parameter arithmetic. Six new
full-GD segment replays match recorded scalar histories exactly. Historical
trajectory reconstruction fails qualification and is explicitly excluded from
claims of reproduction.

The [figure review](../../../../../analyses/027-2026-10-30-minibatch-noise/FIGURE_REVIEW.md) covers readability and scientific limits.
This is a three-initialization characterization of one objective at one LR,
not a general stability criterion or an estimate of larger-model noise.
