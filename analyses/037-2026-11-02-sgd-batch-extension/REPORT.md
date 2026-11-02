# Three-case SGD batch-size and starting-state comparison

Status: complete. All 348 new continuations, six reference trajectories and
both helpers pass raw-file and numerical acceptance. The publication figure
and 16-page manuscript pass numerical and visual review.

Display follow-up: the publication figure now shows only mean adjacent cosine
in one row of three case panels, with both starting states. Retention remains
in the verified results and this report. The earlier two-row publication view
is preserved at commit093b81c; no measurements or sampling have changed.

## Comparison and coverage

H023; Runs214/216 extend the existing Analysis027 MLP experiment to the
user-confirmed cases CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer. This is three model/dataset cases, not three distinct
datasets. Vary batch size and starting state within each case, matching the
objective, initial tensors, learning rate, 200-update continuation horizon,
parameter scope and reduction conventions. Constant rates are 0.02/0.1/0.2;
their selection preceded these batch-size outcomes.

The design has three initialization blocks, eight repetitions per stochastic
condition and one full-GD control per block, from initialization and after
2,000 GD updates. ResNet samples whole fixed 1,000-image normalization groups
at 1/5/20/50 groups. Transformer samples 8/32/128/512/59,680 prepared sequences;
its unbiased gradient estimator divides by the fixed complete-token count,
not a random sampled-token count. Complete endpoint losses include all 50,000
ResNet images or 2,088,600 Transformer supervised tokens. MLP uses its existing
5,000-image objective and batches 32/128/512/2048/5000.

The verified primary coverage is 150 new ResNet plus 198 new Transformer
continuations, and six fresh 2,000-update references. Reusing 198 unchanged MLP
continuations gives 546 trajectories in the figure. Two initial full-GD controls
are the exactly captured first 200 reference updates, with separate full-loss
evaluations; this preserves coverage while avoiding 400 redundant updates.
An additional 200-update GB10/L40S replay bridge is not a primary replicate.
The resulting production execution count is 81,400 updates, excluding all
mechanics/calibration work. The reference policy, control derivations and
runtime qualification are explicit in the linked execution records.

## Observations

The smallest trained-state batch has less negative mean cosine than full GD
in every case, while all trained-state retention means remain below 0.05:

| Case | Smallest batch | Cosine: smallest / full | Retention: smallest / full |
| --- | ---: | ---: | ---: |
| MLP | 32 images | -0.384 / -0.991 | 0.0303 / 0.0319 |
| ResNet-32 | 1,000 images | -0.612 / -0.967 | 0.0349 / 0.0396 |
| Transformer | 8 sequences | -0.258 / -0.993 | 0.0416 / 0.0475 |

Initial-state behavior differs. Full-GD mean cosine is 0.994/0.999/-0.827 for
MLP/ResNet/Transformer; the Transformer enters reversal within 200 updates.
The sampled MLP at batch 32 instead has negative mean cosine(-0.141), showing
that moving SGD can change the regime, beyond fixed-state noise attenuation.

For the Transformer at batch 8, the two starting states yield nearly identical
mean adjacent cosine but different cumulative retention and opposite mean
complete-objective loss changes:

| Starting state | Mean cosine | Retention R_200 | Training-loss change |
| --- | ---: | ---: | ---: |
| Initialization | -0.259496 | 0.109896 | -3.505137 |
| After 2,000 GD updates | -0.258445 | 0.041632 | +0.064266 |

The loss-change sign holds for each initialization's eight-repetition mean.
Full GD improves the trained-state Transformer loss in all three blocks.
Thus an adjacent-angle summary does not rank cumulative movement or training
progress, even within the same model, objective, learning rate and batch size.
Retention weights step magnitudes and all temporal interactions; stochastic
sampling also changes the subsequent parameter path.

## Reduction, uncertainty and limits

Each continuation contributes the mean of 199 defined sampled-gradient cosines
and retention over all 200 gradients. Average repetitions within initialization,
then weight the three initialization blocks equally. Shading spans those three
block means; it is neither a confidence interval nor uncertainty over all
models/datasets. [cells.csv](cells.csv) retains the block values and
[scientific_summary.json](scientific_summary.json) records the panel reduction.

The largest absolute difference between nominal-gradient retention and
actual-update retention over the 348 new forks is4.873e-8. Maximum per-step
relative error of the nominal increment is5.726e-4; these are different audit
quantities. They do not establish bitwise trajectory reproducibility.

Parameters/raw gradients are FP32, geometry reductions FP64 and TF32 is off.
Cloud measurements use PyTorch 2.8/CUDA 12.8; Transformer physical accumulation 512
was qualified before its separately frozen Run216 production. ResNet uses
L40S/RTX4090, with seed 2's 32 smallest-group continuations on GB10/PyTorch 2.11/
CUDA 13 from hash-identical cloud checkpoints. Its initial 200-update full-GD
bridge has endpoint parameter relative L 2 error2.36e-7 and actual-retention absolute
difference4.28e-7. This bounds that measured bridge, not every stochastic path
or long chaotic replay. Batch and runtime effects are not fully separated in
that seed block; shading also combines initialization with runtime variation.

Equal update budgets imply unequal data exposure. The three cases also differ
in objective, data size, architecture and selected rate. These results do not
isolate architecture/data effects, choose an optimal batch size, measure noise
covariance or establish stability/generalization. No new HVP measurement was
made. Complete-objective endpoint loss is a training outcome, not held-out
generalization. Adjacent cosine is not an all-lag interaction statistic.

## Evidence, reproduction and execution

Frozen configs, source/data/initialization hashes and the run-local engine are
in [Run214](../../runs/214-2026-11-02-sgd-batch-extension/README.md) and
[Run216](../../runs/216-2026-11-02-sgd-transformer-microbatch512/README.md).
The complete raw inventories remain local under ignored run artifacts.
Seven mathematical/sampling/recovery tests, numerical runtime qualification,
exact sampling-index replay, independent scalar reduction, selected CPU vector
audits and actual endpoint-displacement checks qualify the measurement.
Worker acceptance requires raw bytes/SHA verification before deletion.

Run the full reduction from the repository root:

```text
python analyses/037-2026-11-02-sgd-batch-extension/02_verify_results.py runs/214-2026-11-02-sgd-batch-extension/artifacts/attempt-01/resnet-seed0 runs/214-2026-11-02-sgd-batch-extension/artifacts/attempt-01/resnet-seed1 runs/214-2026-11-02-sgd-batch-extension/artifacts/attempt-04/resnet-seed2 runs/216-2026-11-02-sgd-transformer-microbatch512/artifacts/attempt-04/transformer-seed0 runs/216-2026-11-02-sgd-transformer-microbatch512/artifacts/attempt-01/transformer-seed1 runs/216-2026-11-02-sgd-transformer-microbatch512/artifacts/attempt-01/transformer-seed2
python analyses/037-2026-11-02-sgd-batch-extension/15_verify_cohort.py
python manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py
python manuscript/draft/revisions/2026-11-02-feedback/02_build_and_verify.py
```

Archive the superseded MLP-only draft figure as recorded in the manuscript
archive manifest before building. Historical Analysis027 source data/artwork
remain unchanged. Figure generation requires a passed 348-fork cohort;
the manuscript verifier independently checks the raw panel reductions,
displayed example, equations, tables, source hashes, archive and compiled assets.
Visual review is a separate final step and is reset on rebuild.

Runpod setup, calibration, retries, storage and retrieval share the user's
US$40/six-hour ceiling, beginning 13:34:56 UTC and ending 19:34:56 UTC on
2026-11-02. Spark assistance uses an isolated directory and preserves its old
suspended queue. Final provisional cost, task-resource readback and retained
Spark storage belong in [closeout.json](closeout.json).

One calibration-only archive was lost after a failed transfer; no production
ran on that worker. The timing snapshot remains and the deletion gates were
strengthened. See [retrieval-incident.json](retrieval-incident.json); successful
production retrieval does not erase this calibration limitation.

Closeout: all 13 task Pods/disks deleted,0 GB retained cloud storage, provisional
US$26.30 including calibration/retries/disk and 4.663 hours wall time. Spark retains
1,378,522,451 bytes on owned storage and has no active task measurement process.
The old suspended queue remains stopped. See closeout.json for exact amounts.
