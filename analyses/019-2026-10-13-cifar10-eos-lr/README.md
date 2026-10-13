# Analysis019: CIFAR-10 full-batch GD/Adam learning-rate ablation

Status: complete. All eight Spark runs reached 20,000 fixed updates; results and
figures are retrieved, hash-verified and numerically audited. See the
[results report](REPORT.md), [observation](observations/001-learning-rate-sweep.md),
[verification](verification.json) and the [current figures](#figures). Run058 is unchanged.

## Comparison

The user selected the owned DGX Spark, the 5k tanh MLP task, GD plus Adam, and a
fixed 20,000 updates per run. H018 is exploratory: how do loss, raw-gradient and
optimizer geometry relate to ordinary/preconditioned Hessian curvature as LR
changes? No directional prediction or universal stability/convergence proxy is
assumed. There is one matched seed 0 initialization; rates are not individually
tuned and this does not establish an optimizer ranking or generalization law.

| Optimizer | Runs | Learning rates |
| --- | --- | --- |
| GD | 059-062 | 0.004, 2/300 (0.006666666667), 0.01, 0.02 |
| Adam | 063-066 | 0.00005, 0.0001, 0.0002, 0.0004 |

All runs use the first 5,000 official CIFAR-10 training images in fixed order,
normalization from all 50,000 training images, and complete 10,000-image test
evaluation every update. The 3072-200-200-10 tanh MLP has 656,810 parameters and
uses half-squared L2 output error to one-hot labels, averaged over images. Frozen
seed 0 initial tensors and data hashes are shared exactly. Full-batch gradients
accumulate five physical batches of 1,000 before one update. Precision is FP32,
TF32 disabled, with FP64 diagnostic reductions and vector prefix sums.

Adam uses betas (0.9, 0.99) and epsilon 1e-7. Both optimizers have zero decay, no clipping,
no schedule, no augmentation and no injected gradient noise. Training continues
through 20,000 updates even if 99% accuracy is reached, so all curves share a
fixed update horizon. Every update uses the same objective; this is not SGD.

## Coverage and interpretation

The [Run058 contract](../../runs/058-2026-10-13-cifar10-eos-calibration/README.md#diagnostics-and-capture-contract)
defines raw gradients, stored momentum m, unbiased momentum mhat, adaptive
directions and true displacements. Norms, adjacent cosine, A/B/B-over-A, squared
geometry, energy, exact all-pair/lag-one/remainder, cross-stream alignments/norm
ratios, parameter norm, net displacement, gradient-dot-update and eta-kappa are
logged at every accepted update. GD has raw and displacement streams only;
Adam includes all five. The underlying raw curves are retained without smoothing.

The two largest algebraic objective-Hessian eigenvalues are measured on all 5,000
training images at state 0, every 100 updates and state 20,000: 201 spectra/run.
Adam also measures two eigenvalues of P^(-1/2)HP^(-1/2) at every positive
curvature state, with P=(1-beta1^t)*(sqrt(vhat)+epsilon), and exposes conventional
D scaling. Each eigenpair has a fresh relative residual check (<= 0.005).
The initial solve uses tolerance 1e-3, ncv 20 and maxiter 200; a predeclared
retry uses ncv 60 and maxiter 400. The
retry handles nonconvergence/residual failure without changing parameters,
objective, requested eigenvalues or tolerance. Solver attempts are recorded.
LA differs from the released code's implicit LM selection and avoids confusing
a large negative eigenvalue with maximum curvature. No dense Hessian is formed.

Train loss/gradient for update t correspond to theta[t-1]; moments and actual
displacement belong to update t, and test evaluation to theta[t]. Curvature
records explicitly name their state index. The gradient-derived kappa on Adam
remains a trajectory proxy; ordinary GD 2/eta is not an Adam stability threshold.
Finite-horizon patterns and local spectral references do not establish a global
stability or convergence theorem. Per-image energies/intra-batch alignment and
arbitrary long-lag profiles are not recovered from scalar mean-gradient logs.

## Execution and recovery

The detached, sequential queue ran on 2026-10-13 from 12:37 to 13:29 UTC
(09:37-10:29 Brasilia), with persistent progress, loss, throughput, memory,
curvature diagnostics and ETC. Actual training/diagnostics took 52.03 minutes;
the initial calibration projected 58 minutes. The declared bounds were 45 minutes
per run and four hours for the queue. No rental or cloud billing. The task retains
1.930 GB on Spark, within its 5 GB allowance, including transfer archives;
prepared Run058 inputs remain separately. Dense raw vector histories are disabled.
Initial, rolling 1,000-update recovery and
final states include optimizer/diagnostic/RNG state and identity hashes.

Source files were promoted from frozen Run058 into src/cifar_eos because this
second experiment reuses them. Run058 is unchanged. The only training-behavior
change is enabling the user-selected fixed horizon; the pre-update finite check
and residual-checked eigensolver retry protect mechanics. All mathematics and a
three-update full-shape smoke pass. Smoke data are not scientific evidence.

Code/configs are frozen in the [source inventory](source_inventory.json); the
packet passes local/remote SHA256. The source inputs and existing isolated
Torch/CUDA13/SciPy runtime are reused read-only from the task-owned calibration
directory. All 120 retained files pass local size/SHA256 checks.
The [final device/storage check](compute_closeout.json) confirms Spark is idle.
No unrelated Spark resources were managed.


## Figures

Current: [cifar10_gd_diagnostics.pdf](figures/cifar10_gd_diagnostics.pdf), a
six-panel GD-only comparison of all four learning rates: unsmoothed linear
training loss, adjacent cosine, eta-kappa, eta*lambda_max, full-prefix raw B/A,
and unsmoothed log-scale raw-gradient norms. Cosine and eta-kappa use complete
100-pair trailing means. See [Observation002](observations/002-gd-six-panel.md),
[numeric checks](gd_figure_manifest.json) and
[visual verification](gd_visual_verification.json).

Adam counterparts: [full 20,000 updates](figures/cifar10_adam_diagnostics.pdf)
and [first 2,000 updates](figures/cifar10_adam_diagnostics_early.pdf). Each uses
ten panels, inserting momentum adjacent cosine, momentum B/A, momentum norm
and the raw-gradient/momentum norm ratio into the corresponding GD sequence.
Momentum means bias-corrected mhat; the norm ratio is raw/mhat, unsmoothed.
Both views share Y scales and the same complete-prefix/window reductions.
See [Observation003](observations/003-adam-ten-panel.md),
[numeric checks](adam_figure_manifest.json) and
[visual verification](adam_visual_verification.json).

The four original detailed PDFs are archived unchanged:

- [cifar10_eos_comparison.pdf](figures/archive/cifar10_eos_comparison.pdf)
- [cifar10_gd_geometry.pdf](figures/archive/cifar10_gd_geometry.pdf)
- [cifar10_adam_geometry.pdf](figures/archive/cifar10_adam_geometry.pdf)
- [loss_and_norms_by_learning_rate.pdf](figures/archive/loss_and_norms_by_learning_rate.pdf)

[Archive identities](archive_manifest.json) preserve the original size/SHA256
and map former paths to current locations. The original figure manifest and
visual verification remain historical records. No run inputs or results changed.

Regenerate the current figure from the retained verified reductions with
`python analyses/019-2026-10-13-cifar10-eos-lr/09_plot_gd.py`, then render/check
with `python analyses/019-2026-10-13-cifar10-eos-lr/07_pdf_qa.py --simple-gd`.
After inspecting the render, add `--accept-reviewed` to record the visual check.
For both Adam views, use script `10_plot_adam.py`, then script `07_pdf_qa.py`
with `--simple-adam`; use the same review procedure.
Scripts 05 and 06 are the original detailed-figure/record generators; the new
figures are generated by scripts 09 and 10. No training is needed for these revisions.

Closeout: 52.03 minutes training/diagnostics; ETC 0; all 160,000 updates and 4,816 eigenpairs audited. Full scalar/checkpoint evidence remains locally and on the owned Spark; no rental/billing. See the report for actual storage and the final device check.
