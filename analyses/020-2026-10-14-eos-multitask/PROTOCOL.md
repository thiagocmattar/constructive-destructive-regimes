# Comparison and measurement contract

## Question and publication scope

H019 asks whether consecutive-gradient alignment, projected gradient change and
complete-prefix displacement track the GD spectral regime across the three
user-selected EOS cases, and how fresh mini-batch sampling changes those measures.
The intended venue is IEEE TNNLS. A robust empirical basis needs replicated
initializations/sampling and explicit counterexamples, not selected curves from
one seed. The cases cover two datasets and three architectures; architecture,
loss, data size and task change together across cases. No architecture effect can
be isolated by this cross-case comparison.

Use seeds 0,1,2, frozen before execution, with paired initial tensors and identical
objectives across full and mini-batch conditions within each case. Initialization
varies across seeds. Sampling draws vary independently across updates; batch
conditions share prefixes of each seed's random permutations. Probe RNG is
separate and cannot perturb training. No augmentation, dropout, clipping,
momentum, decay or LR schedule. All parameters/gradients are FP32, TF32 disabled;
global scalar reductions and prefix vectors are FP64. Loss is averaged over
images for classification and supervised tokens for the Transformer.

Provisional task rates, before production outcomes: MLP eta=0.01, ResNet
eta=0.2, Transformer eta=0.1, all within the respective paper demonstrations.
Budgets/cadences are fixed from calibration before launching production, and
will not be shortened or extended based on attractive diagnostic outcomes.
The existing MLP seed0 eta=0.01 Run061 can be reused only after verifying identical
initialization, objective and settings; it is not a new independent seed.

## Objective units and batch sampling

MLP units are the first 5,000 CIFAR images, with the released code's normalization.
ResNet units are 50 **fixed groups** of 1,000 channel-normalized CIFAR images.
BatchNorm operates inside each group, exactly as in the paper's ghost objective.
Varying the number of sampled groups changes gradient noise while preserving the
objective; varying BN group size would change the objective and is excluded.
Transformer units are fixed, causal, 35-token sequences from the full prepared
WikiText-2 corpus. The historical tutorial's 20 training streams produce 59,680
sequence units and 2,088,600 supervised tokens. Its eight trimmed stream-tail
tokens and 20 initial context tokens are explicitly declared, not silently
discarded. Short final chunks are padded only for batching; padded targets are
excluded. Validation/test cover all prepared supervised tokens, with separately
recorded tutorial stream-tail/context exclusions.

For N objective units and full objective F=(1/N)sum_i ell_i, the raw SGD gradient
is the average of b units sampled uniformly **without replacement within each
update**, independently across updates. For language chunks, ell_i is rescaled
by N / total_supervised_tokens. Thus random short-chunk counts do not change the
denominator or introduce sampling bias. A full-batch update accumulates every
objective unit before one SGD step.

Fixed update budgets compare optimization dynamics, not equal data exposure.
Figures also expose sampled images/tokens or objective units. A supplementary
view over the common sampled-example budget is descriptive; it cannot substitute
for the missing long-horizon full-batch dynamics. Frozen-parameter controls below
isolate sampling effects without any trajectory/data-exposure confound.

## Diagnostics

Every update records the unsmoothed batch objective estimate, raw global norm,
adjacent cosine rho, A=sum||g||, B=||sum g||, B/A, vector energy, exact all-pair
and lag-one interactions, actual update geometry, and the FP32 displacement
identity residual. The first cosine and zero-denominator quantities are null,
with reasons; no denominator epsilon is added. Gradients are recorded before
the plain SGD update. Full-prefix sums never reset at a plot boundary.

q_t=eta*kappa_hat_t=1-g[t-1]^T g[t]/||g[t-1]||^2 is the gradient-only projected
change. Under SGD it is not automatically a maximum Hessian eigenvalue. With
stochastic gradients it includes a realized noise-difference contribution and
must not be relabeled 'curvature' without correction.

At declared consecutive-pair probe steps, independently compute full gradients
mu[t-1] and mu[t] at exactly the two parameter states, xi=g-mu, and
epsilon=g[t-1]^T(xi[t]-xi[t-1])/||g[t-1]||^2. Record q+epsilon and verify its
identity with g[t-1]^T(mu[t-1]-mu[t])/||g[t-1]||^2. These pairs are **adjacent
optimizer updates**, not adjacent sparse log rows. Full-objective losses are
measured at probe states; unsmoothed stochastic batch losses remain distinct.

Ordinary sharpness is the largest **algebraic** Hessian eigenvalue, using
matrix-free HVPs, with independent eigenpair residual <=0.005. The plot uses
eta*lambda_max and a GD quadratic reference at 2. MLP uses all 5k examples;
ResNet uses the first five fixed ghost groups, following the paper's subset;
Transformer uses an explicitly declared fixed sequence subset. A subset Hessian
is not the Hessian of the complete training objective. No dense Hessian is formed.

## Gradient noise, causal controls and uncertainty

For unit gradients h_i, define S^2=sum_i||h_i-mu||^2/(N-1). The conditional
batch-gradient noise variance trace is V_b=(1/b-1/N)S^2. This is vector-gradient
variance, not variance of batch loss. MLP per-image squared gradient norms are
calculated exactly using layer outer-product identities over all images;
ResNet unit covariance uses every fixed group; Transformer uses an unbiased
sample covariance estimate from distinct uniformly sampled sequences. Coverage,
sampling seed and precision are logged. Its sampled variance has estimation
uncertainty and is not presented as an exact population quantity.

Frozen-state experiments use independent real-data batch gradients at saved GD
parameter states, without updating parameters. They measure adjacent cosine,
norms and normalized accumulation at matched sequence lengths across batch
conditions, with repeated sampling sequences. Full batch gives identical
nonzero gradients, cosine=1 and B/A=1 at a frozen state. Stochastic trajectories
need not show a monotone B/A relationship: SGD can change the learned objective
region and signal-to-noise ratio. Report seed-level values, paired differences
and intervals. Three seeds are a practical minimum, not proof of universality.

## Execution, retention and stopping

Use owned Spark only, without rental costs. Freeze all production configs/source
hashes before first updates. Detached queue, persistent logs, bounded process
deadlines, periodic progress/ETC/memory and numerical-failure warnings. Keep
initial, selected intermediate, rolling recovery and final checkpoints with
optimizer, RNG, data-sampling and diagnostic state. Scalar histories cannot
recover arbitrary missing gradient lags. Dense training-gradient history is
disabled; frozen probes retain sufficient individual measurements for uncertainty.
Retrieve artifacts with relative-path byte-size and SHA256 inventory verification.
Do not remove or modify unrelated Spark experiments.
