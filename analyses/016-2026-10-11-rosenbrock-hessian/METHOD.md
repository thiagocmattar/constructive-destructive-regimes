# H015: Rosenbrock curvature along the Analysis013 recipe

Pre-execution contract, 2026-10-11. The user requested a Rosenbrock run like
Analysis013 with the largest Hessian eigenvalue at every point, described as
critical curvature. This operationalizes that phrase as the largest algebraic
eigenvalue of the unpenalized objective Hessian, without positing an AdamW
stability boundary or a directional hypothesis.

Runs055-057 repeat Runs038-040: start (0,1), f=(1-x)^2+100(y-x^2)^2, exact
analytic gradients, CPU float64, 2,000 updates, constant LR 0.01 /
sqrt(0.001*0.01) / 0.001, zero initial moments, betas (0.9,0.95), epsilon
1e-8 outside the square root, decoupled decay 0.1 on both coordinates, no
clipping, no early stopping. LR is the sole varied numerical factor; the new
diagnostic never feeds back into updates. There are no data, seeds, validation
split, cloud resources or bills. This fixed-budget diagnostic replication
preserves the earlier immutable attempts.

For every saved theta[t], t=0,...,2000, retain the full float64 symmetric matrix

    H(x,y) = [[2 + 1200*x*x - 400*y, -400*x],
              [-400*x,              200]]

and both ordered eigenvalues from numpy.linalg.eigvalsh. Lambda_max is the
larger algebraic eigenvalue, not the largest absolute eigenvalue. Lambda_min
is retained to show that lambda_max does not disclose negative curvature.
Hessian/loss at t>0 describe the post-update point; the raw gradient used by
update t is evaluated at theta[t-1]. Units are objective per coordinate squared.
Decoupled decay is not an L2 term in f or H. No Hessian smoothing, sampling,
thinning, weighting, epsilon or denominator convention is needed. The full
positions permit exact curvature reconstruction; compute it after numerical
trajectory generation within the same run, before writing its immutable history.

Retain every original raw/momentum/adaptive/displacement vector, second moment,
all prefix geometry, loss and post-gradient norm, plus final optimizer state.
First adjacent cosine and B/A at A=0 remain undefined. Keep the original joint
stationarity criterion (f<=1e-8 and gradient norm<=1e-4; final 100 positions),
without early stopping. Progress events persist every 200 updates.

Before acceptance: objective/gradient mechanics and every optimizer update
against the independent Analysis010/PyTorch checks; bitwise replay; bitwise
identity of all old arrays with matched Analysis013 controls; complete Hessian
coverage against SciPy rosen_hess at every point, autograd and finite differences
at additional points, and a closed-form 2x2 eigenvalue check at every point;
config/source/launcher and inventory size/hash integrity. Test generalized b,
indefinite start H(0,1)=diag(-398,200), and optimum H(1,1)=[[802,-400],[-400,200]].

Two publication PDFs: a focused trajectory/lambda_max/loss/alignment view and
the original nine diagnostic panels plus lambda_max and lambda_min. Curvature
is unsmoothed; only adjacent cosines use full finite 100-update trailing means,
first at update101. Use the same LR colors and first strict y-minimum events
as Analysis013, at their original position indices; independently check the
rendered series, lines, rolling windows and PDF pages, then inspect PNG renders.
All figure sources and captions belong to this analysis observation.

The existing three-run cohort simulated/saved in about0.184 seconds and stored
2.84 MB; these are planning references. Extra 2x2 diagnostics are small; measured
runtime and storage will be recorded. No billable authorization is needed.
Histories/logs/scratch remain ignored and locally retained. Shared implementation
is new, since all three runs need it; existing numerical/plot modules are unedited.

Interpretation is descriptive for one deterministic function/start/optimizer/
horizon. Lambda_max measures local maximum directional curvature. A plain
gradient-descent quadratic threshold cannot be directly applied to this AdamW
trajectory, whose first/second moments and decoupled decay change its dynamics.
No stability, causality, generalization, asymptotic convergence or LLM transfer
is established by curvature, alignment sign or norm alone.
