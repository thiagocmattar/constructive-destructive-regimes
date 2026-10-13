# CIFAR-10 full-batch feasibility and timing

Implemented and verified on the user's RTX5070Ti Laptop(12GB) and DGX Spark
GB10(128GB unified memory). Both completed24updates of GD and24updates of Adam
from identical frozen inputs. No full scientific run or cloud rental was launched.

The setup is the confirmed first5,000 CIFAR-10 training images,3072-200-200-10
tanh MLP,656,810parameters, half-squared L2 loss averaged over images, physical
batches of1,000 accumulated into one full gradient. Initialization is seed0;
normalization uses the complete50,000-image training split. There is no
augmentation, clipping, decay, LR schedule, TF32 or added gradient noise. Test
loss/accuracy cover all10,000 test images every update.

GD uses LR0.01. Adam uses exploratory LR1e-4, betas(0.9,0.99), epsilon1e-7;
the Adam settings follow the released adaptive-EOS example. These are different
step scales, not a tuned optimizer ranking. The proposed first horizon is20,000
updates or99% training accuracy, whichever comes first; the official GD demo's
100,000-update ceiling is also shown as an extrapolation below.

## Measured timings and ETC

Each method/device has20post-warmup update samples. Updates include full test
evaluation and our streaming diagnostics. There are only two ordinary Hessian
solves and one preconditioned solve per Adam path, all in the initial24updates.

| Device | Method | Median update | Peak allocated / reserved | Projected20k | Planning allowance20k | Approximate100k ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX5070Ti Laptop | GD |11.51ms |257.2/274MiB |5.2min |10.7min |26min |
| RTX5070Ti Laptop | Adam |15.28ms |341.6/374MiB |8.0min |17.1min |40min |
| DGX Spark GB10 | GD |12.84ms |257.5/274MiB |6.2min |12.5min |31min |
| DGX Spark GB10 | Adam |19.66ms |341.6/374MiB |8.4min |15.4min |42min |

Projections include top2ordinary Hessian eigenvalues every100updates,
initial/final spectra, and checkpoints. Adam additionally measures top2
preconditioned eigenvalues at each post-initial curvature state. The planning
allowance uses1.3times the empirical p90 update timing,3times the observed
curvature cost and60seconds of fixed overhead. This is a stated scenario,
not a probabilistic bound. Convergence time, late eigensolver iteration counts,
sustained thermals and plotting overhead have not been measured. Larger budgets
can stop earlier at99% training accuracy; reaching that goal is not guaranteed.

The two20k methods together project13.2min on RTX and14.6min on Spark; allow
about30min for the pilot pair. RTX is a practical first choice for this small
MLP. These samples do not establish a general hardware speed ranking.

Hessian cadence matters: evaluating both spectra every single update projects
about2.3hours for RTX GD and4.9hours for RTX Adam at20k, even before extra
late-solve allowance. The every100update recommendation preserves per-update
geometry while keeping curvature overhead manageable. The eigen solver uses
matrix-free products; a dense float32 Hessian would require approximately1.57TiB.

## Diagnostics and memory

The recorder includes unsmoothed full training/test loss and accuracy; parameter
norm and true net displacement; raw-gradient, stored-momentum m, bias-corrected
momentum mhat, adaptive-direction and actual-update norms/cosines/A/B/B-over-A;
vector alignment and norm ratios; squared prefix geometry, exact all-pair energy,
lag-one terms and their remainder; gradient-dot-update loss prediction; and
the noise-free eta-kappa proxy. m and mhat are separate streams. GD has no
momentum/adaptive stream. See[the full contract](README.md#diagnostics-and-capture-contract).

The paper's top2ordinary Hessian eigenvalues, eta*lambda_max and GD2/eta
reference are recorded. Adam also has the released code's symmetrically
preconditioned Hessian and conventional D scaling. LA(largest algebraic) is
declared explicitly; the released utility defaults to LM(largest magnitude).
Every returned eigenpair passes a relative residual check. Ordinary curvature
and the GD threshold are not treated as Adam stability criteria.

Peak CUDA allocation is below0.35GiB with all configured diagnostics. These
allocator figures exclude CUDA driver context, desktop applications and CPU
ARPACK workspaces; reserving about2GB of free VRAM is a comfortable operational
allowance for this exact task. Normalization preparation uses several GiB of
system RAM;16GB system RAM is a comfortable target. Spark memory is ample.
These memory measurements do not apply to full50k VGG/ResNet experiments.

Scalar logs and recoverable accumulators are retained. Optional dense raw-gradient
history is disabled:20k updates would add52.54GB per method;100k would add262.72GB.
Arbitrary subsequent temporal profiles and per-sample decompositions require
additional histories/measurements. Scalar logs alone do not supply them.

## Verification, provenance and retained storage

- Five independent mathematical check groups pass, including short-chunk
  normalization, direct prefix sums, dense Hessian/HVP/eigenvalue comparison,
  Adam bias correction, true displacements and checkpoint continuity.
- All96calibration updates, eight ordinary spectra and two preconditioned
  spectra pass scalar/coverage/residual checks. Final checkpoint states
  independently match momentum/direction vectors and telescoping displacements.
- Twenty Spark files(126.34MB), input packet and result archive pass size/SHA256
  verification. The original CIFAR archive passes its canonical MD5;
  preprocessing agrees bitwise with the pinned released functions on11probes.
- Source revision0d8ceeb3ad98e28930762205934f0a941faeb479, tensor hashes and
  download identity are recorded in[inputs.json](inputs.json). Both devices
  loaded identical initial tensors. Runtime/CUDA architecture differ;
  bitwise cross-hardware replay is not assumed.
- About844MB is retained locally and696MB on Spark, including inputs, transfer
  archives, checkpoints and the task-local SciPy dependency. All large files
  stay outside Git. The shared Spark environment was preserved. GPU is idle;
  no cloud resources, rental charges or resource cleanup are outstanding.

Generating/reducing scripts:00_prepare.py,01_experiment.py,02_verify.py,
03_benchmark.py,04_report.py and05_accept.py. Raw timing/eigenvalue evidence is
inartifacts/attempt-01-rtx andartifacts/attempt-01-spark; the derived scenarios
are in[calibration_results.json](calibration_results.json), with
[verification](verification.json) and[retrieval verification](retrieval_verification.json).
