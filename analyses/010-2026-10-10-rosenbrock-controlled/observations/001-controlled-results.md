# Observation001: controlled objective and alignment outcomes

Question: how do first-moment memory and decoupled decay change objective
navigation and adjacent vector geometry on a known 2D objective?

Sources: every history.npz and summary.json under Runs028-031,
artifacts/attempts/001-local. Exact source SHA-256, sizes, config/source identities
and scientific completeness are in each run's inventory.json, manifest.json,
verification.json and this analysis's verification.json.02_plot.py generates
figures/controlled_comparison.pdf plus one trajectory_geometry.pdf in each run.
figure_manifest.json freezes source and output identities. Per-run observations
own the latter PDFs; all eight PDFs passed visual review using ignored renders.

Coverage: identical start (-1.2,1), f=24.2, all 10,000 updates of each trajectory;
objective includes position 0. No discarded tail, failed/skipped update, or
sampled gradient. Exact gradients and all vector/prefix reductions use float64.
Main comparison loss/norm/trajectory panels are unsmoothed. Cosine panels use
a trailing 100-update mean requiring 100 defined pairs; the first visible mean
is therefore at update 101. The interactive explorer shows unsmoothed values.
Full-domain contours use500×500 analytic meshes; the static final close-up
uses the separately declared500×500 domain/levels in02_plot.py.

Caption: **Controlled Rosenbrock, momentum × decoupled decay.** Four trajectories
share objective/start/LR/beta2/epsilon/horizon. Top panels show full navigation
and a close-up of the valley with final positions. Remaining panels show
post-update objective, trailing raw/momentum cosine means and raw gradient norm.
Run028/029/030/031 use blue/orange/green/purple with solid/dashed/dash-dot/dotted
lines; markers distinguish start, terminal position and known minimum.
Gradient cosine at t compares gradients at positions t-2,t-1, while displayed
loss/position is at t. No claim of instantaneous geometric effect is implied.

Result: final loss is3.45215e-6/0.0224507/0.000973108/0.0826439 for Runs028-031.
Momentum improves the fixed-horizon objective at each matched decay value;
decay worsens it at each matched momentum setting. The beta1=0 controls still
use adaptive second moments, so this is not a comparison against plain SGD.
Only Run028 ever meets the joint f<=1e-8 and gradient norm<=1e-4 test, first at
7,678; all four fail the final 100 sustained-update criterion. Its last 1,000 loss
range2.21917e-9..1.42761e-4 reveals persistent oscillation near(1,1).

Run028 raw/momentum negative-pair frequencies are 80.638%/80.358% among 9,999
defined adjacent pairs. Its momentum late median is -0.999999905. Run030's
late median is +0.632115509 while its objective floor is higher. In Run028,
4,543 transitions have negative raw cosine and decreased objective, and 1,549
have positive raw cosine and increased objective. These simultaneous counts
are descriptive and use the declared gradient/transition indexing; they are
not predictive tests, independent samples, or evidence of cosine mediation.

Uncertainty/limits: one deterministic start at one LR/beta2 in 2D; no stochastic
seed confidence intervals. Oscillatory final values are phase-sensitive, which
is why best loss, final loss, last 1,000 range and sustained thresholds are all
retained. No formal stability, asymptotic convergence, LLM transfer,
generalization, or information-storage claim. Initial/final tensors, raw
histories and complete optimizer moments remain local. Manuscript unchanged.
