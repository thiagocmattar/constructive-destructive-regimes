# Compact backward-lag manuscript view, 2026-11-02

`figures/gd_backward_compact.pdf` extracts the middle row of the complete
GD figure into a 1x3 publication view. The manuscript copy is under
`manuscript/draft/figures/analysis035/`. It retains every exact point for
lags 1--256, with no smoothing, parity reduction, endpoint averaging or
compression. Sources are passed `verification.json` and `results.json`;
all 768 plotted values independently match `lag_profiles.csv`.

Selection: the same middle retained starting state (2000) and endpoint
state (2256) in all three models, at their previously chosen largest GD
rates 0.04/0.1/0.2. The manuscript defines g_t at parameter state t-1,
so this endpoint is gradient index t=2257, not t=2256. The caption makes
this conversion explicit. No result-based endpoint selection occurs.

Observation: a small cosine at one separation need not imply loss of
directional structure at greater lags; opposing and reinforcing values
reappear. Amplitudes vary among models and endpoints, and this view does
not establish monotone decay or repeated traversal of a particular minimum.
As an all-endpoint check, each of the nine GD profiles has both signs
within lags 193--256; exact ranges are in the revision figure manifest.

Coverage/uncertainty: one pair per lag, one original initialization per
model, and dependent checkpoint forks. No confidence intervals. These are
separately recomputed 256-update continuations; identical initial tensors
do not imply bitwise equivalence to the original 3,000-update trajectories.
The substantial later replay differences documented in the analysis remain
explicitly qualified in the manuscript appendix. This is not an all-pair
interaction estimate, stability test or generalization result.

Generator: `manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py`.
Caption: `manuscript/draft/sections/04-gd-experiments.tex`.
Provenance and verification: the revision's `figure_manifest.json`,
`verification.json` and `review.json`. No training or remote work.
