# Comparison and measurement contract

H020 asks how changing the constant GD learning rate changes objective loss,
maximum measured curvature and projected gradient change across the three EOS
tasks. This extends H019 without changing its objective, architecture or
diagnostics. The user explicitly authorized extra runs on Spark on 2026-10-15.

| Case | Learning rates | Full update objective | Hessian objective |
| --- | --- | --- | --- |
| CIFAR-10 / tanh 200-200 MLP | 0.004, 0.01, 0.02 | First 5,000 images, half squared error | All 5,000 images |
| CIFAR-10 / ResNet-32 | 0.1, 0.2, 0.4 | All 50,000 images, mean cross entropy over 50 fixed 1k ghost groups | First five fixed groups (5,000 images) |
| WikiText-2 / Transformer | 0.05, 0.1, 0.2 | All 2,088,600 prepared target tokens in 59,680 causal sequences, token-mean NLL | First 2,500 prepared sequences |

Use 500 fixed updates and seeds 0,1,2 for every learning rate. Initial tensors
are identical across learning rates within each seed and case. Other factors
are fixed: data preparation/order, physical accumulation, precision, diagnostic
coverage, stopping horizon and optimizer. Plain full-batch GD has no momentum,
decay, clipping, schedule, augmentation or dropout. The frozen Analysis 020 input
manifest pins the data and historical model sources. Its protocol documents
normalization, token exclusions, ghost normalization and differences from the
original EOS implementation. No hyperparameter selection uses test outcomes.

Reuse the Analysis 020 GD runs at the middle learning rate. Reuse the first 500
updates of Analysis 019 Runs 059 and 062 for MLP seed 0 at the other rates, after
checking initialization/data/objective parity. These are repeated uses of the
same evidence, not independent replications. Run IDs 104-119 contain the sixteen
new conditions. The previous 20k MLP horizons are truncated only for plotting;
their raw evidence is preserved. All larger models receive actual new updates
at each new learning rate, not rescaled curves from the middle rate.

The reused legacy MLP runs computed the two largest algebraic eigenvalues with
a larger Krylov space; the newer runs request only the largest. This figure
uses the largest value in both cases, at the same states and with the same
residual threshold. The older runs also evaluated test loss more often.
These diagnostic implementation differences do not alter deterministic MLP
updates, but the reused measurements are not a bitwise matched eigensolver
replication. Their scalar and objective-parity checks remain required.

Record full training loss before each update and at the final state, without
temporal smoothing. Every adjacent pair records raw-gradient cosine and norm;
the plotted quantity is exactly

    q_t = eta*kappa_hat_t = 1 - rho_t*||g_t||/||g_(t-1)||
        = 1 - g_(t-1)^T g_t / ||g_(t-1)||^2, t=2,...,500.

It is undefined at the first pair or a zero denominator, with no numerical
epsilon added. For deterministic GD this is a finite-step projected gradient
change; its interpretation as local Hessian directional curvature has a Taylor
remainder. It is not the maximum eigenvalue. ReLU nondifferentiability can also
limit a smooth local interpretation. Noise epsilon is zero in this comparison.
Pair t spans states theta_(t-2), theta_(t-1); plot it at the ending state t-1.

Largest **algebraic** Hessian eigenvalues are measured at parameter states
0,100,...,500; multiply each by its own run's learning rate. Fresh relative
residual must be <=0.005. Lines connecting spectral markers are guides, not
unmeasured intermediate spectra. Subset Hessians for ResNet and Transformer
cannot be treated as the exact full-objective maximum curvature. Reference
lines at eta*lambda=2 and q=1 are respectively the GD quadratic reference and
the exact gradient-reversal boundary for this recorded q, not global stability
guarantees. All panels use linear axes and no temporal smoothing.

For each LR, plot the mean of three seeds, with faint individual seed traces.
Do not smooth oscillations or pool seed traces in time. Three seeds provide
descriptive replication, not a confidence interval. The 500-update horizon is
the matched first stage and need not reach the asymptotic edge-of-stability
regime. Architecture, task, objective and dataset size differ across columns.
No claim about generalization, universality or causal mediation follows.

Execution: reuse unchanged training code and mechanical/mathematical checks
from Analysis 020. Run sequentially on owned Spark after that queue terminates.
Measured complete runs give about 2h per ResNet and 4.45h per Transformer:
about 39h additional compute, with a 1.4x planning allowance and a 72h execution
deadline. The predecessor has its own 72h deadline. A waiting process allocates
no GPU. Allow 25 GB additional retained evidence on each machine; available disk
was 227 GB local and 3.7 TB remote at preparation. No rented compute or storage.

Persist attempt-local sources, logs, progress, recovery/final checkpoints,
optimizer/RNG/diagnostic state and transfer inventories. Poll every five
minutes. Do not interrupt unrelated or preceding runs. On numerical failure,
retain the failure and stop publication; do not drop a rate/seed or substitute a
short prefix. Validate transferred file sizes and SHA-256 hashes before plotting.
Generated publication files remain awaiting visual review until their current
renders have been inspected. Do not overwrite concurrent manuscript edits:
insert the figure into the current TeX only after all evidence passes, preserve
all other bytes, and require its rendered position to follow Table 1.
