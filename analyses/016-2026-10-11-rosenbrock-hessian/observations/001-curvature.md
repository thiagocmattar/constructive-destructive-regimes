# Observation001: Curvature along matched Rosenbrock trajectories

H015 asks for largest Hessian eigenvalue along the Analysis013 recipe. Runs055-057
match038-040 at LR0.01/midpoint/0.001; all original arrays are bitwise identical.
[Method](../METHOD.md), [cohort](../cohort.json), [results](../results.json),
[scientific checks](../verification.json), [figure checks](../figure_verification.json).
Each results row identifies its source directory, history SHA-256 and size;
the owning inventories identify all raw/compact artifacts. Sources and script
hashes also appear in each figure manifest. Reduction: ../03_record.py;
plotting: ../02_plot.py, with the locally frozen extended layout ../plot_full.py.

Coverage: all 2,001 saved positions per rate, including t=0 and t=2000, and all
2,000 optimizer vectors. H uses both coordinates of the unpenalized objective;
eigenvalues are float64 symmetric eigvalsh, checked independently at every point.
No curvature/loss filtering, smoothing, weighting, sampling or thinning.
Only raw/momentum adjacent cosines use 100-update trailing arithmetic means of
complete finite windows, first at 101 (1,900 displayed means/stream/rate).
The gradient used by update t is at theta[t-1]; H and loss at t use theta[t].

| LR | Final loss | Final lambda_max | Peak lambda_max (update) | lambda_max at y-turn | Final lambda_min |
| --- | ---: | ---: | --- | ---: | ---: |
| 0.01 | 0.00097226306 | 953.119491261 | 954.678167901 (1434) | 553.500307134 | 0.371290830 |
| 0.003162278 | 0.000117958119 | 984.994949568 | 985.607388325 (1797) | 515.654138297 | 0.379700391 |
| 0.001 | 1.10471357e-05 | 996.329309332 | 996.412281325 (1972) | 495.743726917 | 0.404253617 |

Lambda_max starts at 200 for all rates, despite lambda_min=-398: the starting
Hessian is indefinite. Curvature rises along valley traversal and approaches
the optimum's lambda_max=1001.600639361. Large LR arrives sooner and has an
early overshoot/dip in curvature; its late plateau is lower. The smallest LR
has the largest final lambda_max and the lowest final loss in this horizon.
This is a positional association in matched trajectories, not a curvature
intervention. Max curvature does not by itself identify loss quality.

The first y-turn indices remain 81/228/655. Their lambda_max values are in the
table; these geometric events are not curvature extrema or stability thresholds.
The first nonnegative lambda_min positions are
66/211/634.
Negative lambda_min covers 82/211/634 saved positions respectively; an initial
sign crossing does not imply curvature stays nonnegative thereafter.
All three fail the original joint loss/gradient stationarity criterion.

Caption - [focused figure](../figures/rosenbrock_hessian.pdf): **Exact Hessian
curvature along Rosenbrock AdamW trajectories.** Start (0,1); 2,000 updates;
betas (0.9,0.95), epsilon 1e-8, decoupled decay 0.1, CPU float64. (a) Complete
trajectories over objective contours, square start, star optimum, open final
points and diamond first y-turns; domain includes every trajectory point.
(b) Unsmoothed largest algebraic Hessian eigenvalue at each saved point; dotted
gray reference is its value at (1,1), not an optimizer threshold. (c) Objective
loss on log Y. (d) 100-update adjacent-cosine means, raw solid and stored momentum
dashed. LR colors match Analysis013; temporal first-y-turn references are dashed.

Caption - [complete diagnostics](../figures/rosenbrock_hessian_diagnostics.pdf):
**Analysis013 geometry with exact curvature.** (a) All trajectories on the
original full-domain contours; (b) loss; (c,d) 100-update raw/momentum adjacent
cosine means; (e) vector norms; (f,g) raw/momentum A=sum of norms and B=norm of
sum; (h,i) B/A; (j,k) largest/smallest algebraic Hessian eigenvalues at all
positions. Loss uses log Y; all other temporal axes are linear. All panels
retain LR-colored first-y-turn references at 81/228/655. Decay adds no L2 term
to the objective or Hessian. The first cosine and B/A at A=0 remain undefined.

Uncertainty/limits: one deterministic function/start/optimizer/horizon; no
sampling uncertainty or confidence bands. LR scales both adaptive descent and
decay. A quadratic plain-gradient-descent curvature boundary cannot be directly
applied to AdamW's evolving moments/preconditioner. No causal alignment benefit,
universal stability threshold, asymptotic convergence, generalization or LLM
transfer follows. Complete histories are locally retained and ignored.
New raw storage 3,105,699 bytes; simulation/saving 0.195s.
No cloud resources, rental or storage billing.
