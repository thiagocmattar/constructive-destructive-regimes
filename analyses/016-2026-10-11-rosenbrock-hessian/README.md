# Analysis016: Rosenbrock with exact Hessian curvature

Completed and verified, 2026-10-11. **[Focused curvature figure](figures/rosenbrock_hessian.pdf)**;
[complete eleven-panel diagnostics](figures/rosenbrock_hessian_diagnostics.pdf).
Repeat Analysis013 from (0,1), at the same three LRs and 2,000 updates, measuring
the largest objective-Hessian eigenvalue at every saved position. New Runs055-057
retain every original diagnostic and both eigenvalues/full 2x2 Hessians.
All original arrays are bitwise identical to matched Runs038-040.

| LR | Final loss | Final lambda_max | Peak lambda_max (update) | lambda_max at y-turn | Final lambda_min |
| --- | ---: | ---: | --- | ---: | ---: |
| 0.01 | 0.00097226306 | 953.119491261 | 954.678167901 (1434) | 553.500307134 | 0.371290830 |
| 0.003162278 | 0.000117958119 | 984.994949568 | 985.607388325 (1797) | 515.654138297 | 0.379700391 |
| 0.001 | 1.10471357e-05 | 996.329309332 | 996.412281325 (1972) | 495.743726917 | 0.404253617 |

Curvature starts 200 and rises toward the optimum's 1001.60. Smallest LR has the
largest final curvature and lowest final loss here. First y-turns 81/228/655
are retained in both figures. Lambda_min starts -398, so maximum curvature alone
does not describe the initial indefinite geometry or an AdamW stability boundary.

Matched recipe: exact Rosenbrock f=(1-x)^2+100(y-x^2)^2, AdamW betas (0.9,0.95),
epsilon 1e-8, decoupled decay 0.1, zero initial moments, no clipping, constant LR,
CPU float64. Hessian/loss cover 2,001 positions t=0..2000; gradient for update t
uses theta[t-1]. Curvature is unsmoothed; only adjacent cosines use complete
finite 100-update trailing means, first at 101. All fail joint stationarity.

[Method](METHOD.md), [observation/captions](observations/001-curvature.md),
[results](results.json), [scientific verification](verification.json),
[figure verification](figure_verification.json). Full replay, independent AdamW
for 6,000 updates, SciPy Hessians/closed-form eigenvalues at 6,003 positions,
additional autograd/finite-difference checks, source/artifact sizes/hashes and
actual rendered-array identities pass. PDF layout review is recorded in the
figure verification receipt. Histories 3,105,699 bytes, locally retained;
simulation/saving 0.195s; no cloud/billing.

Reproduce reductions with `python analyses/016-2026-10-11-rosenbrock-hessian/01_verify.py`,
then 02_plot.py and 03_record.py at this path. 00_prepare.py and owning 01_run.py
refuse existing inputs/attempts; do not execute into completed runs.
One deterministic setup establishes descriptive geometry, without sampling
uncertainty, an AdamW threshold, causal curvature benefit or LLM transfer.
