# Run058: CIFAR-10 Edge of Stability implementation and calibration

Status: implementation verified and both hardware calibrations complete/retrieved.
[Timing, memory and proposed setup](REPORT.md), [measured scenarios](calibration_results.json),
[verification](verification.json). Calibration only; no full scientific experiment
or cloud rental is launched by this run.
The user requested local feasibility, GD/Adam support, our geometry diagnostics,
the paper's Hessian measurements, memory and ETC. The user confirmed the paper's
5,000-example tanh MLP with GD and Adam. VGG/ResNet training is not implemented here.

## Matched inputs and varying factors

The sample unit is one CIFAR-10 image. Use the first 5,000 official training
images in fixed order. Normalization uses all 50,000 training images, reproducing
the released code: divide uint8 pixels by255, compute coordinate-wise means,
center, then compute population standard deviations on centered data. The
normalization is per spatial/channel coordinate, not three channel-only means.
There are no augmentations or additional stochastic gradients. Test evaluation
covers all10,000 images and never controls stopping.

The network is3072-200-200-10, with tanh hidden activations and all biases.
Its656,810 parameters are initialized once using PyTorch's CPU defaults with
seed0; identical frozen tensors are loaded on both devices and for both methods.
Squared loss is half the squared L2 output error to a one-hot target, averaged
over images, with no division by10 classes. Cross-entropy is also implemented
and mathematically tested, but not used in this calibration. The physical batch
is1,000; gradients are divided by the full5,000-example denominator and summed
before one optimizer update. The dataset and test tensors reside on GPU.

The varying factors are optimizer and hardware: GD at LR0.01 versus exploratory
Adam at LR0.0001, betas(0.9,0.99), epsilon1e-7. Both have constant LR, zero
decay, no clipping and no mixed precision/TF32. Adam's settings follow the
official adaptive-EOS README example, rather than the Adam script's beta2
default0.995. Different LRs do not establish a matched-step or tuned optimizer
comparison. Calibration is24updates/method/device, with the first4 excluded
from update-time estimates; this is mechanics evidence, not a scientific result.

The proposed first production budget is20,000updates/method, or99% training
accuracy reached earlier. It is a proposal, not the original demonstration's
100,000-update ceiling. Full test evaluation is every update; ordinary Hessian
measurements are at initialization, every100updates and the final state. Adam
also has preconditioned eigenvalues after its moments exist. Checkpoints occur
initially, every1,000updates (latest recovery) and finally. No production budget
has been executed. [Config](config.json) freezes the calibration recipe.

## Diagnostics and capture contract

Every accepted update records the unsmoothed full training loss and accuracy,
test loss/accuracy with explicit state index, parameter norm and net displacement,
gradient-dot-update first-order loss prediction, vector alignment/norm ratios,
and the restatement's noise-free eta-kappa proxy. Training loss/gradient are
evaluated at theta[t-1]; moments and actual displacement belong to update t.
Test evaluation is at theta[t], a one-step offset explicitly recorded in each row.

For each stream record global norm, adjacent cosine, A=sum(norms), B=norm(sum),
B/A, A squared, B squared, triangle gap, energy=sum(squared norms), exact all-pair
interaction=B squared-energy, exact lag-one interaction and their difference.
Whole-model coordinate reductions and vector prefix sums use float64; parameters,
raw gradients and actual Adam moments use float32. Vectors are detached and
measurements never mutate optimizer state. Undefined values are JSON null with
reasons; no artificial denominator epsilon is added.

GD has raw-gradient and actual-displacement streams. Adam additionally has:

- Stored first moment m[t], matching existing Pythia/Rosenbrock momentum panels.
- Bias-corrected mhat[t]=m[t]/(1-beta1^t), reported separately.
- Adaptive direction d[t]=mhat[t]/(sqrt(vhat[t])+epsilon), excluding LR.
- Actual displacement theta[t]-theta[t-1], including float32 parameter rounding.

Kappa is(1-dot(g[t-1],g[t])/norm(g[t-1])^2)/eta. There is no injected noise,
so the noise correction is zero. On Adam it remains a gradient-trajectory proxy,
not a certified Hessian eigenvalue or stability criterion. Comparing scalar
geometry with Hessian curvature does not identify per-sample energy/intra-batch
alignment, which cannot be recovered from a mean gradient.

The paper's ordinary top2eigenvalues use matrix-free full-objective Hessian-vector
products and SciPy ARPACK/Lanczos. This implementation selects largest algebraic
eigenvalues(LA), explicitly differing from the released utilities' implicit
largest-magnitude(LM) selection; the latter can include a large negative mode.
Change the declared config toLM for that code-level variant. Eigensolver tolerance
is1e-3, ncv20; each eigenpair has an independently computed relative residual
and must be within0.005. Failed solves abort; missing diagnostics are not silently
accepted. No dense656810-square Hessian is formed.

For Adam, also measure P^(-1/2) H P^(-1/2), using the released code's
P=(1-beta1^t)*(sqrt(vhat)+epsilon). The conventional D-preconditioned eigenvalues
are provided by exact scalar rescaling, where D=sqrt(vhat)+epsilon. Ordinary
and preconditioned spectra are labeled separately. The GD2/eta reference is
never treated as an Adam stability threshold.

Scalar histories and recovery accumulators are retained. Optional raw vector
history is supported but off: it costs2,627,240bytes/update, or52.54GB for20,000
updates per method. Dense m/d histories, arbitrary later lag profiles and
per-example gradients are not implicitly reconstructed from scalar logs. Initial,
latest recovery and final checkpoints include optimizer, diagnostic states,
random states, index and scientific-input hashes. Resume writes a new output
directory and explicitly labels the continuation's starting index.

## Commands

```powershell
python runs/058-2026-10-13-cifar10-eos-calibration/00_prepare.py --mirror
python runs/058-2026-10-13-cifar10-eos-calibration/02_verify.py --output runs/058-2026-10-13-cifar10-eos-calibration/mathematical_verification.json
python runs/058-2026-10-13-cifar10-eos-calibration/03_benchmark.py --inputs cache/cifar10-eos --output runs/058-2026-10-13-cifar10-eos-calibration/artifacts/attempt-01-rtx
```

Production uses01_experiment.py without--calibrate and a new owning run/config.
Resume additionally passes--resume path/to/recovery.pt. No output directory may
be reused. Spark calibration runs detached with a persistent queue, logs and
15-minute timeout per method; completed files are inventoried and verified after
retrieval. Both devices are user-owned; no rental/billable service is used.

Sources: [original paper](https://arxiv.org/abs/2103.00065),
[official repository](https://github.com/locuslab/edge-of-stability/tree/github),
[MIT's unchanged archive mirror](https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python).
The archive is verified against its original MD5; SHA256, source revision,
normalization/data tensors and common initialization are pinned ininputs.json.
Downloaded reference source copies remain local outside Git.

Spark's existing Torch runtime initially lacked SciPy. The failed preflight
occurred before creating an attempt or taking an update; its log is retained
remotely. SciPy1.18.1 was installed under this task's dependencies/ directory,
without changing the shared environment; the unchanged calibration then passed.
The isolated dependency directory must be included in PYTHONPATH on Spark.
