# Observation001: adjacent and anchored raw gradients under cold Adam

Sources: Run205 retained start checkpoints and original gradients; Runs209/211's
five-rate calibration; Run210's full-objective production records; Analysis033
GD anchored profiles. Source/input identities, complete scalar reductions and
transfer receipts are linked from the report. Generator: `02_analyze.py`.

Reduction: each cosine is a full-vector FP64 dot divided by Euclidean norms,
checked on GPU and independently on CPU; no projection, smoothing or epsilon.
Calibration equal-weights primary checkpoints. Last 64 summaries use raw adjacent
pairs; anchored half-gap is the median of 32 complete odd/even half differences.

Coverage/uncertainty: 27 dependent forks from one seed 0 initialization per model,
25 primary, two flagged historical reconstructions. All three original anchors;
individual 256/512/2048 horizons match GD. No seed error bars or independent
LR-tuning evaluation. Training objective only, no validation/generalization claim.

Figure captions:

- `calibration.pdf`: final 16 mean training loss / initial loss after 64 Adam
  updates at five rates. Thin lines are primary retained states; the black curve
  is their geometric mean, equivalent to the mean-log selection score. Red
  crosses fail at least one declared safeguard. The dashed vertical line is the
  common model rate selected by the declared loss/safeguard rule.
- `anchored_profiles.pdf`: exact cos(g0,g[k]) along Adam (solid), with the
  previously measured GD reference (faint dashed) at identical initial states
  and horizons. Colors identify parent GD histories. Horizontal coordinate is
  symlog with linear region through 16; raw values, no smoothing.
- `adjacent_profiles.pdf`: exact cos(g[k-1],g[k]) along the new Adam trajectories,
  at every full-objective optimizer boundary. Same source colors, anchors and
  horizons; the first adjacent pair is k=1. Symlog horizontal coordinate.
- `adjacent_first128.pdf`: the same adjacent pairs restricted to 1..128 on a
  linear horizontal coordinate to expose the optimizer-reset transient.

Trajectory panels mark the two unqualified historical starts [U]; these never
select LR or enter primary summaries. Declining anchored odd/even gaps and
positive local alignment are descriptive geometry, not a convergence/stability
test or proof that Adam outperforms GD. Raw full gradients differ from actual
Adam moments, adaptive directions and parameter increments.
