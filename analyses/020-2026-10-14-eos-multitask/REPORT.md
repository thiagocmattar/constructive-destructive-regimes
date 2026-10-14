# Analysis020: first verified results and production execution budget

Status: source review, mathematical verification, all three production-shaped
smokes and owned-Spark calibration complete. Run068 frozen-state control is
complete, retrieved/hash verified, numerically checked and visually reviewed.
The 500-update first GD/SGD stage is launched and running; its results remain pending. The longer paper-length horizon remains a subsequent decision. These control results are not a completed EOS/TNNLS reproduction.

## What the original paper tests

The three user-selected cases use two real datasets: CIFAR-10 with a tanh MLP,
CIFAR-10 with ResNet-32, and WikiText-2 with a small word-level Transformer.
The paper also tests VGG-11 and synthetic tasks; it does not test ViT.
[Exact practical setup and source qualifications](SOURCE_REVIEW.md).
ResNet's effective full update covers 50,000 images, accumulated over 50 fixed
ghost groups of 1,000. Its Hessian uses the first 5,000 images. The MLP uses5,000
images; the complete prepared language objective covers 2,088,600 supervised tokens.
All tail/context exclusions inherited from tutorial stream preprocessing are
declared in Run067 inputs. No small training subset replaces the language corpus.

## Real-data gradient estimation noise

Run068 took 37.05 minutes on owned Spark; rental cost zero. Three initializations
per case, 32 independent 128-draw sequences per batch condition, 36 conditions and
147,456 recorded vector observations. At each fixed initial parameter vector,
noise is the actual sampled batch-gradient error against the complete gradient.
ResNet samples whole ghost groups to preserve its objective. No parameter updates are applied in this real-data sampling control.
Normalized B/A in this experiment is gradient accumulation, not actual motion.

| Case | Smallest batch | Relative noise RMS | Mean adjacent cosine | Mean B/A |
| --- | ---: | ---: | ---: | ---: |
| mlp | 32 images | 1.232 | 0.398 | 0.639 |
| resnet | 1000 images | 0.403 | 0.863 | 0.930 |
| transformer | 8 sequences | 1.071 | 0.467 | 0.687 |

These values average seeds 0, 1, 2 and 32 repetitions within each seed. The publication
figure spans the range across initializations; that range is not a 95% confidence
interval. Repeated sampling quantifies conditional Monte Carlo uncertainty, not
variation over datasets or a universal training law. Greater measured noise reduces the frozen positive alignment and accumulation in these conditions.
The smallest available ResNet ghost batch retains substantial alignment.
Approaching zero requires sufficiently large noise relative to the signal.
The gradient-only projected-change statistic can be nonzero without parameter
motion. Its measured noise correction removes that contribution up to roundoff.

## Training plan and ETC

The [protocol](PROTOCOL.md) pairs three seeds and frozen initial tensors within
each case. Plain GD/SGD, constant LR, no decay/clipping/momentum/schedule/augmentation.
Rates 0.01/0.2/0.1 for MLP/ResNet/Transformer; three stochastic batch sizes per
case, plus full-batch controls. Every raw adjacent pair and full-prefix norm,
energy, interaction and B/A is logged. Complete-gradient paired probes provide
the actual noise correction; vector covariance estimates include population
coverage. Ordinary largest-algebraic Hessian eigenvalues have independent
residual checks. Sparse reference/Hessian measurements are explicitly labeled.

The suite has 35 new training runs plus one verified existing seed0 MLP control.
The current [500-update plan](staged-500-plan.json) estimates
29.2 hours, with a40.9-hour allowance
and72-hour hard stop. It is a first stage, not full paper-length evidence.
The [longer plan](paper-length-plan.json) uses20k/2k/15k updates for the respective
cases and estimates20.2 days, qualified to
28.2 days. Its21-day hard cap does not cover the entire
uncertainty allowance; a longer occupation budget would need a concrete choice.
Both keep complete datasets, bounded detached processes, persistent progress,
initial/intermediate/recovery/final states and a40GB retained-storage allowance.
No rental or cloud service is authorized/used. Early calibration does not measure
late eigensolver clustering, validation overhead drift or other users' device load.

## Paper artifacts and limits

[Publication control figure](figures/eos_frozen_noise_comparison.pdf),
[figure/array manifest](frozen_figure_manifest.json),
[observation](observations/001-frozen-real-data-noise.md).
The formulation report moves the noise discussion after normalized displacement,
adds finite-population noise variance and the fixed-parameter accumulation limit,
and includes this control with its sampling/trajectory distinction. Existing
GD and Rosenbrock evidence remains unchanged. Training loss, directional-change,
sharpness and displacement comparisons across all three cases remain pending.

Three seeds at one prespecified LR per case are a replicated foundation; they
do not guarantee journal readiness. Claims must remain conditional on the tested
objectives, batch semantics, precision and horizons. Batch-size sweeps change
data exposure at a fixed update budget. Frozen controls isolate noise; moving
trajectories may reverse the ordering because signal and curvature change.
ReLU/BatchNorm cases also require care interpreting a finite gradient difference
as local Hessian curvature. No generalization claim follows from these diagnostics.
