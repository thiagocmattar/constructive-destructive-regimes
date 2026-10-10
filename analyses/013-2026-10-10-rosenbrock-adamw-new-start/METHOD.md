# H012: Rosenbrock AdamW from a new starting point

Pre-execution contract, 2026-10-10. The user requested the Analysis012 method
from another point and selected **(0, 1)**. No directional prediction is made.
The question is how the objective trajectory and raw-gradient/momentum
geometry change when only the initial point changes from (-1.2, 1) to (0, 1).
This is one additional deterministic starting point, not a distribution of starts.

Run038/039/040 use constant learning rates 1e-2, sqrt(1e-3*1e-2) =
0.0031622776601683794, and 1e-3, respectively. Pair them with completed
Runs035/036/037 from [Analysis012](../012-2026-10-10-rosenbrock-adamw-lr/README.md).
Within the new cohort vary only LR; across each old/new pair vary only the
initial point, excluding descriptive run metadata. Config/source/history
hashes and matching of all other inputs are verified before comparisons.
Old attempts, receipts, histories, figures and numerical source stay preserved.

Use the unchanged shared [numerical implementation](../../src/controlled/rosenbrock.py):
f(x,y)=(1-x)^2+100(y-x^2)^2, optimum (1,1), analytic full 2D gradient,
CPU float64, no clipping, zero initial first/second moments, betas (0.9,0.95),
epsilon 1e-8 outside the square root, decoupled weight decay 0.1 on both
coordinates, constant LR from update 1, exactly 2,000 updates without early
stopping. There is no LR schedule, L2 penalty in the objective, dataset,
sampling, seed, validation split or cloud resource. Initial objective is 101,
versus 24.2 at the old start; horizons are matched in updates, not initial loss.

For update t, g_t=grad f(theta_(t-1)); m_t is the stored, uncorrected Adam
first moment; d_t is the bias-corrected/preconditioned Adam direction excluding
LR and decay; u_t=theta_t-theta_(t-1) is actual displacement including decay.
Capture all 2,001 positions/losses/post-gradient norms and all 2,000 vectors,
second moments and per-prefix metrics for g/m/d/u at every update, preserving
complete NPZ histories and final optimizer state under each owning run.

For each stream z: A_t=sum_(s<=t)||z_s||, B_t=||sum_(s<=t)z_s|| and ratio B/A.
Adjacent cosine compares z_(t-1) and z_t. First cosine and zero-vector pairs
are undefined; A=0 gives undefined B/A, while positive A with B=0 gives zero.
No epsilon, clipping or weighting is added to diagnostics. Retain energy,
exact all-pair interaction and lag-one terms as in the preceding analyses.
The descriptive stationarity criterion is f<=1e-8 and post-gradient norm<=1e-4,
with the final 100 positions required jointly for the sustained check. Keep
the fixed budget regardless of criterion hits; no asymptotic claim is intended.

Reuse the latest Analysis012 figure: one 16-by-23-inch PDF, full-domain
contour/trajectory panel above a four-row/two-column grid. Show start, global
optimum, all trajectory positions and final points. The landscape includes all
paths without zoom; the former domain is retained unless expansion is needed.
(b) Loss on log Y. (c,d) Trailing 100-update arithmetic means of raw/momentum
adjacent cosine, requiring 100 finite consecutive values, so the first mean
is at update 101 when only the first cosine is undefined. (e,f,g) Vector norms,
raw A/B and momentum A/B on linear Y starting at zero. (h,i) B/A on linear Y.
Keep the LR colors/styles and all unsmoothed histories/statistics; no thinning
or uncertainty bands. Plot markers/labels must use the actual stored start.

Before acceptance, verify analytic objective/gradient against SciPy, autograd
and finite differences; replay every recorded array exactly; check every
optimizer update using independent PyTorch AdamW fed the captured gradients;
check all-stream geometry, zero-denominator behavior, momentum prefix identity,
decay displacement and telescoping; verify artifact sizes/SHA-256. Check every
rolling mean against direct windows, scales, PDF text/page integrity and the
rendered layout. Generate reductions and observations with old/new paired
comparisons; update registries and commit completed work without publishing.

Local execution is already authorized by the user's request. Prior same-budget
cohort took about 0.176 seconds for simulation/saving and retained about 2.91 MB
of histories; these are qualified planning figures, not new measurements.
Verification and plotting add seconds. No billable execution is needed. Frozen
attempts refuse overwrite; progress events persist every 200 updates. Retain
all local evidence, with histories/scratch/logs ignored by Git.

Interpretation: differences reflect initialization sensitivity of this fixed
AdamW recipe, not independently manipulated alignment or decay. Initial loss,
gradient direction and decay displacement change with the start. Positive
cosine or a particular B/A does not establish better loss, stability,
generalization, a universal LR ranking or transfer to LLM training.
