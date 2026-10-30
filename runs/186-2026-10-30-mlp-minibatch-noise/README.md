# Run186: MLP sampling noise and SGD displacement

H023 / [Analysis027](../../analyses/027-2026-10-30-minibatch-noise/README.md).
Completed 2026-10-30 on one task-owned Runpod RTX 4090. Production duration:
1,371.56 seconds. Scientific source, configuration and input manifest are frozen.

CIFAR first 5,000 training images, tanh 200-200 MLP, constant LR0.02,
initialization seeds 0/1/2, trained states 0/1,000/2,000, batches
32/128/512/2,048/5,000. Half-squared error is summed over outputs and averaged
across examples. Uniform subsets are independently resampled across updates;
batch sizes share permutation prefixes. No clipping, momentum, decay, schedule,
augmentation or dropout. FP32 parameters/gradients, FP64 geometry, TF32 off.
Full gradients, losses, covariance traces and HVPs use all 5,000 training images.
No test/validation performance or stability claim is part of this comparison.

The 45 frozen cells contain 32 sequences of 128 draws each, including repeated
deterministic controls. Moving cells contain eight 200-update sampling
repetitions per stochastic batch and one deterministic full-batch control.
Totals: 184,320 frozen observations, 297 forks, 59,400 continuation updates,
6,000 reference updates and 1,188 directional HVP pairs. Frozen accumulation
is not actual parameter motion.

All scalars, sampled indices, reference/probe/final parameter states and rolling
recovery with RNG and diagnostic accumulators are retained locally under
`artifacts/attempt-01/`. Dense per-step gradient vectors are not retained.
Production evidence comprises 7,068 inventory-listed files (14.917 GB), all
size/SHA-256 verified before cleanup. Environment logs are in `execution/`.

Historical Run062 reconstruction fails the predeclared scalar tolerances and
is reported separately. All new comparisons share the same new saved checkpoint;
six full-GD continuation segments exactly replay their own new references in
loss, norm and cosine. All recorded denominators are nonzero. The independent
[audit](../../analyses/027-2026-10-30-minibatch-noise/verification.json) verifies
indices, state matching, raw coverage, identities and stored-parameter motion.

Run185 supplies mechanics/calibration evidence, not paper results. Analysis027
owns the final [report](../../analyses/027-2026-10-30-minibatch-noise/REPORT.md),
publication figures, uncertainty and interpretation. Task cloud resources are
deleted; the whole task's provisional compute/storage estimate is USD0.87.
Spark and unrelated resources were untouched.
