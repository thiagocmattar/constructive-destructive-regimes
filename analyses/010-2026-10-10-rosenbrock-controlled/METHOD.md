# Controlled Rosenbrock pilot: H009

User authorization, 2026-10-10: implement and execute a first controlled
Rosenbrock experiment, carry Analysis009's metrics into it, and show contour
navigation alongside objective decay and adjacent alignment. "Analysis 00:09"
is resolved as Analysis009. The direction is user-selected; the following
small local factorial is an exploratory implementation choice, with no
predeclared direction or alignment threshold for successful optimization.

## Comparison frozen before execution

Use f(x,y) = (1-x)^2 + 100(y-x^2)^2. Its unique zero and global minimum is
(1,1). Start all conditions at (-1.2,1), with f=24.2. Matched factors are the
objective, exact analytic gradients, start, 10,000-update horizon, LR=0.01
constant from the first update, beta2=0.95, epsilon=1e-8 outside sqrt, CPU
float64, parameter order (x,y), and no clipping. Vary beta1 in {0,0.9} and
decoupled weight decay in {0,0.1} in four independent runs (028-031).
Beta2 and positive beta1/decay retain the earlier Pythia values. LR is a pilot
choice, not a claim that it is optimal for Rosenbrock. Beta1=0 removes the
first-moment memory but retains adaptive second-moment preconditioning.
Decay=0 is Adam as the zero-decay special case of AdamW. Both coordinates
receive decay. No batch, data, noise, validation set, tokenizer or scientific
random seed exists in this exact deterministic setup.

At update t=1..10,000, g_t = grad f(theta_(t-1)); moments start at zero:

    m_t = beta1*m_(t-1) + (1-beta1)*g_t
    v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
    d_t = [m_t/(1-beta1^t)] / [sqrt(v_t/(1-beta2^t)) + epsilon]
    theta_t = (1-LR*decay)*theta_(t-1) - LR*d_t
    u_t = theta_t - theta_(t-1)

g is raw and unclipped; m is the actual uncorrected moment, d is the adaptive
direction excluding LR, descent sign and decay; u is the actual displacement
including all three. The run-local NumPy implementation is checked against
torch.optim.AdamW with identical captured gradients for all 10,000 updates.
Analytic gradients are checked against SciPy, autograd and finite differences.

## Measurement and indexing

At every update retain theta before/after, objective before/after, raw g, m,
v, d, u; all are float64. For each z in {g,m,d,u}, retain individual norm,
A_t=sum ||z_s||, B_t=||sum z_s||, B/A, adjacent cosine, A^2, B^2 and
A^2-B^2. A^2-B^2 is the nonnegative triangle gap used for comparison with
Analysis009; it is **not** the signed all-pair interaction. Also retain
E=sum ||z_s||^2, I=B^2-E=2*sum_(i<j) dot(z_i,z_j), L1=2*sum dot(z_(s-1),z_s),
and I-L1, exactly from the full history. These quantify lag-one omission
without assuming distant terms vanish. Retain norm ratios m/g, d/g, d/m.

First cosine is undefined (`first_update`). A zero current or previous norm
makes cosine undefined (`zero_vector_norm`); a zero A makes B/A undefined
(`all_zero_vectors`). Zero denominator norm ratios are undefined
(`zero_denominator_norm`). NPZ represents these using NaN only in those metric
arrays; JSON/interactive values use null. Exact cancellation with A>0 has
defined B/A=0. Do not silently clip valid geometry or replace zero norms with
epsilon. Verify mathematical bounds to numerical tolerance.

Plots show the post-update objective f(theta_t) and geometry captured for
transition theta_(t-1) -> theta_t. Cosine at t compares z_(t-1),z_t;
raw cosine therefore compares gradients evaluated at theta_(t-2),theta_(t-1).
Slider t=0 shows initial position/objective with no vector diagnostics.
This indexing avoids assigning a newly evaluated gradient to the wrong point.
Also retain actual objective change, g_t dot u_t (first-order prediction),
post-update gradient norm and distance to the known optimum. Adjacent
constructive/destructive means positive/negative dot product between the named
stream's consecutive vectors, not positive/negative effect on loss.

## Coverage, convergence and limits

Run every condition for all 10,000 updates without early stopping. Convergence
threshold is both f<=1e-8 and ||grad f||<=1e-4; report first simultaneous hit
and whether every one of the final 100 updates passes. A transient small loss
does not meet the sustained criterion. Also report best/final loss and the
last-1000 loss range. This finite test does not prove asymptotic convergence.
Weight decay can oppose reaching Rosenbrock's nonzero optimum; decoupled AdamW
decay does not generally minimize f plus an ordinary scalar L2 penalty.

All measures cover the full objective and every update. No sampling uncertainty
bars or seed-based errors are justified: this is one deterministic start at one
LR and beta2 on one 2D objective. Time statistics are descriptive, with serial
dependence. Momentum and decay interventions change subsequent gradients through
the trajectory; they do not hold a common gradient sequence in the experiment.
The common sequence is used only for independent optimizer verification.
Do not infer generalization, information destruction, formal stability,
universality to LLMs, or causal mediation by cosine. No manuscript is edited.

## Execution and evidence

Local CPU only, no cloud resources or new billing. Estimated execution is
seconds for four trajectories, with a few minutes for independent verification
and plotting. Retain every gradient/vector and second moment in ignored
history.npz; retain compact summaries, hashes, manifests and final state in
Git. Source/config/initial-point hashes are frozen at launch. Progress is
persisted every 1000 updates; all raw arrays are stored at every update.
The short jobs need no detached cloud process or periodic recovery policy;
complete histories and final moments remain available locally. Executed
attempt directories cannot be overwritten. Re-execution of modified scientific
inputs requires new run IDs. The analysis checks sizes/SHA-256 and exact replay
before setting completed status. Scientific source stays in Run028 and is
shared by the four explicitly named configurations; cross-run plotting belongs
to Analysis010.

References: [SciPy Rosenbrock definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html),
[PyTorch AdamW algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
