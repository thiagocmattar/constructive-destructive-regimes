# H011: Rosenbrock AdamW at three learning rates

Authorized by the user's 2026-10-10 request for another experiment like
Analysis011, adding weight decay, dropping LR 1e-4 and adding a middle point
between 1e-3 and 1e-2. Use weight decay coefficient 0.1, the value discussed
immediately before this request. Resolve the middle point on the existing
logarithmic LR scale: sqrt(1e-3 * 1e-2) = 0.0031622776601683794.
Create three new run IDs (035-037) and Analysis012; preserve all prior evidence.
Exploratory question: how do the requested loss/navigation and raw/momentum
geometry behave with decay across this LR interval? No directional prediction.

Match Analysis011: f(x,y)=(1-x)^2+100(y-x^2)^2, start (-1.2,1), zero initial
moments, betas (0.9,0.95), epsilon 1e-8 outside sqrt, constant LR from step 1,
exact analytic gradients, no clipping, both coordinates, CPU float64,
exactly 2,000 updates without early stopping. Only LR varies within the new
cohort; weight decay 0.1 is fixed. There is no data, batching, sampling,
experiment seed or validation split. No new LLM training or cloud resources.

Weight decay is a decoupled parameter shrinkage, not an LR schedule:

    g_t = grad f(theta_(t-1))
    m_t = 0.9*m_(t-1) + 0.1*g_t
    v_t = 0.95*v_(t-1) + 0.05*g_t^2
    d_t = [m_t/(1-0.9^t)] / [sqrt(v_t/(1-0.95^t)) + 1e-8]
    theta_t = (1-LR*0.1)*theta_(t-1) - LR*d_t

The per-step shrinkage factor therefore varies with LR: 0.999 at 1e-2,
0.9996837722339832 at the midpoint and 0.9999 at 1e-3. The plotted objective
and raw gradient remain those of Rosenbrock, with no L2 term added to f/g.
Both x and y receive decay. Momentum excludes shrinkage; adaptive direction
excludes LR/decay; actual displacement includes them. The known optimum of
the measured f remains (1,1), while shrinkage pulls toward (0,0).

Capture the same four complete vector histories and second moments using the
unchanged src/controlled/rosenbrock.py. Objective/point/post-gradient norm:
positions 0..2000; vectors and prefix diagnostics: transitions 1..2000.
Raw g_t uses the previous position; stored m_t is the uncorrected first moment
after incorporating g_t. Cosine compares adjacent named-stream vectors.
At every prefix, A=sum ||z_s||, B=||sum z_s||; use B/A. A=0 gives undefined
B/A; positive A with B=0 gives 0. First cosine and zero-norm adjacent pairs
are undefined. Norm ratios and exact temporal sums remain captured as before.

One figure: the same 16-by-23-inch full-domain contour/trajectory panel above
the four-row/two-column grid: loss, raw cosine, momentum cosine, raw/momentum
norms, raw A and B, momentum A and B, raw B/A, momentum B/A. Every value is
shown; no smoothing/thinning/zoom. Cosines are individual points, matching
Analysis011 and making the comparison independent of the former 100-step
averaging in Analysis010. Preserve common LR endpoint colors/line styles.

Verify all updates against independent PyTorch AdamW, exact replay, SciPy/
autograd/finite-difference objective/gradient, stream prefixes, momentum
identity, displacement with decay, denominator cases and sizes/SHA-256.
LR 1e-2 must match the first 2,000 updates of Run030 exactly. For the two shared
endpoint LRs, compare frozen no-decay Runs032/033 using matched inputs and
verified source hashes. The new midpoint has no executed no-decay control;
do not claim an isolated decay effect there or launch an extra control.

Keep the same convergence criterion: f<=1e-8 AND post-gradient norm<=1e-4,
report first simultaneous hit and whether all final 100 positions pass.
Alignment sign, ratios or small loss alone are not stationarity, stability,
information destruction or a causal mechanism. Fixed horizon, one function/
start/beta2 and no stochastic replicates provide no uncertainty band or
asymptotic/general/LLM result. LR also scales the shrinkage step, and the
interventions change the encountered gradients through their own trajectories.

Local trajectory/save execution is expected to take under a second in total
based on the preceding matched cohort; checks and figures take seconds to
minutes. Retain all raw histories (expected below 10 MB), final state,
progress every 200 updates and hashed config/source/input manifests. No
billable compute, transfer or cleanup is involved. Freeze scientific inputs
before execution; only analysis/presentation records may evolve afterward.
