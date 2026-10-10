# H010: momentum without decay at three learning rates

Authorized by the user's 2026-10-10 refinement: retain only momentum with no
decay; use 2,000 updates; compare LR 1e-2, 1e-3, 1e-4; provide one simplified
figure with a large full-trajectory contour panel and eight metric panels.
"A single run" means one optimizer setting per LR; repository provenance uses
three new run IDs because LR and the executed horizon are scientific inputs.
The prior executed runs and figures are preserved.

Match the previous exact Rosenbrock method: f(x,y)=(1-x)^2+100(y-x^2)^2,
start (-1.2,1), betas (0.9,0.95), epsilon 1e-8 outside sqrt, weight decay 0,
constant LR from update 1, no clipping, CPU float64, order (x,y), zero initial
moments, analytic full-objective gradients. Only LR varies among Runs032-034.
No training data, stochastic sampling, experiment seed, validation split or
batch exists. Match the full 2,000-update budget, not travel distance/wall time.
There is no early stopping or hyperparameter selection after seeing results.

At transition t=1..2000, raw g_t=grad f(theta_(t-1)), m_t is the actual stored
uncorrected first moment after incorporating g_t, d_t is the bias-corrected
adaptive direction before LR/decay, and u_t=theta_t-theta_(t-1). Capture all
four histories as before; the requested figure shows only raw g and momentum m.
Objective points cover positions 0..2000; vector metrics cover transitions
1..2000. Adjacent raw cosine at t uses gradients at theta_(t-2),theta_(t-1),
while the displayed loss is post-update f(theta_t). Keep this indexing explicit.

A_z(t)=sum_(s<=t)||z_s||; B_z(t)=||sum_(s<=t)z_s||. Before handoff, the user
corrected the initial A/B wording to **B/A**. Both ratio panels and reported
values therefore use B/A, matching the existing captured metric. This changes
the reduction/presentation only; executed optimizer inputs and histories are
unchanged. Compute B/A directly with A as denominator. A=0 makes B/A undefined
(`zero_sum_of_vector_norms`); B=0 with positive A is a defined ratio of 0.
Do not insert epsilon. First adjacent cosine is undefined (`first_update`),
and zero adjacent vector norms make it undefined (`zero_vector_norm`). The
first ratio is 1 for a nonzero vector; mathematical 0<=B/A<=1 is verified.
Use all recorded updates without smoothing or thinning in the figure. Maintain
the prior convergence test f<=1e-8 AND post-gradient norm<=1e-4, and distinguish
transient hits from all final 100 positions passing. One deterministic start,
objective, beta2 and fixed horizon do not establish asymptotic convergence,
an optimal LR, information destruction, stability or an LLM result.

Promote the previously verified Run028 implementation into src/controlled now
that another experiment needs it, preserving the original executed file.
The numerical implementation is unchanged; only its shared-location docstring
changes. Freeze shared source/config/initial-point hashes before each execution.
Retain every raw vector and moment in ignored NPZ, progress every 200 updates,
compact inventory/summary/manifest and final state. Short local jobs require
seconds; plotting/verification add seconds to minutes. Estimated raw storage
below 10 MB; no cloud resources or billing and no transfer/cleanup gate.
Verify objective/gradient, independent PyTorch AdamW at every update, exact
replay, prefix geometry, no-decay displacement identity, B/A denominators,
2,000-update coverage and sizes/SHA-256 before completed status. The LR1e-2
trajectory should match the first 2,000 updates of Run028, because only its
stopping horizon changes. This is a reproducibility check, not a new seed.

Figure: one large full-domain contour/trajectory panel spans both columns;
below it, a 4×2 grid in the user's order: loss / raw cosine; momentum cosine /
raw and momentum norms; raw A and B / momentum A and B; raw B/A / momentum B/A.
Three LR colors plus redundant line styles identify trajectories and loss/ratio
series. Adjacent cosines use one point per defined pair with no connecting
lines, avoiding a dense curtain between rapidly alternating +1/-1 values. Solid versus
dashed distinguishes A versus B or g versus m within each condition. Give the trajectory a
separate explicit legend for start, optimum, end and path; label level values
and provide a contour color scale. Increase the landscape's physical area,
retain equal coordinate units, and use one full domain with no zoom panel.
