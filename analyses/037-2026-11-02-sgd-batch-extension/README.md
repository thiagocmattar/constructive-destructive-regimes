# Analysis037: three-case SGD batch-size comparison

H023; Runs214/216. Complete:348 new continuations, six fresh 2,000-update GD
references, and the additional 200-update Spark runtime bridge. Reusing 198
unchanged Analysis027 MLP continuations supplies 546 trajectories for the
requested 2-by-3 figure. All three initializations and eight stochastic
repetitions per condition are retained. The cases are CIFAR-10/MLP,
CIFAR-10/ResNet-32 and WikiText-2/Transformer, not three distinct datasets.

The [report](REPORT.md) gives the comparison, observations and reproduction.
The [figure observation](observations/001-three-case-sgd.md) records sources,
reduction, caption, coverage, descriptive ranges and interpretation limits.
[results.json](results.json), [cohort_acceptance.json](cohort_acceptance.json),
[scientific_summary.json](scientific_summary.json) and [cells.csv](cells.csv)
preserve independently verified repetition/block reductions.

Smaller batches weaken trained-state negative mean cosine across all three
cases, while every trained-state retention mean stays below 0.05. Initial-state
responses differ: full GD is positively aligned for MLP/ResNet but strongly
reversing for Transformer; sampled MLP paths can also change sign. Transformer
batch 8 has nearly equal mean cosine from both starts (about-0.26), but retention
changes from 0.110 to 0.042 and mean full-training-loss change from-3.505 to+0.064.
Adjacent alignment alone therefore does not order cumulative motion or loss.

## Design and verification

Within each case, vary batch size and initial/2k starting state, matching the
objective, pinned initial tensors, constant rate and 200-update horizon. Rates
0.02/0.1/0.2 were selected before these outcomes. ResNet keeps fixed 1 k-image
normalization groups; Transformer uses the unbiased fixed-token-denominator
estimator and full 2,088,600-token objective. See the [measurement contract](../../manuscript/draft/revisions/2026-11-02-feedback/SGD_EXTENSION.md).

Seven CPU math/sampling/recovery checks pass. Every new continuation has 200
updates and 199 defined cosines; exact sampling indices, independent scalar
reduction, selected CPU vector audits, actual endpoint displacement and complete
raw-file byte/SHA inventories pass. Source/data/config/initial tensors remain
unchanged; [executed-inputs-verification.json](executed-inputs-verification.json)
rechecks all 29 manifest entries in each run. Raw evidence remains under ignored
run artifacts, rather than in Git.

Run216 freezes Transformer physical accumulation 512, qualified before
production; its effective sampling batches/objective are unchanged. Cloud
workers use PyTorch 2.8/CUDA 12.8. ResNet uses L40S/4090; seed 2's 32 smallest-group
forks use Spark GB10/PyTorch 2.11/CUDA 13 from hash-identical cloud checkpoints.
The [initial200-update bridge](spark-bridge-acceptance.json) has parameter
relative L 2 error2.36e-7 versus L40S; this is not long chaotic replay validation.
Batch/runtime effects are not separated in that block, and shading also
combines initialization/runtime variation. Equal updates imply unequal exposure;
case differences do not isolate architecture or dataset. No covariance, HVP,
stability or generalization result is inferred from these two rows.

Two initial full-GD controls reuse exactly captured reference prefixes;
348 trajectories cover 69,600 continuation updates but require 69,200 explicit
continuation updates. Adding 12,000 reference and 200 bridge updates gives 81,400
production updates, excluding mechanics/calibration. Spark hands off 32 primary
forks; a qualified extra L40S hands off two original trained-state batch 20
forks. No primary trajectory is duplicated. The bounded queue wait pauses only
the scheduler, leaving its active measurement child running; all handoffs
arrive before that child ends. Receipts preserve each placement and verification.

## Execution and cleanup

The user approved US$40/four hours, then six hours with the same total ceiling,
and requested other GPU classes plus idle Spark assistance. All 13 task Pods
(including calibration retries and the auxiliary worker) are deleted after
appropriate retrieval/acceptance. Retained task cloud storage:0 GB. Fresh account
readback preserves unrelated resources. Provisional elapsed-time GPU plus
conservative disk estimate: **US$26.30**; allocation-to-final-deletion duration:
**4.663 hours**, inside the six-hour ceiling. This is not posted billing.
[closeout.json](closeout.json) contains the full ledger and readback.

Spark has no remaining task measurement process. Its isolated 1.379 GB project
is retained on owned storage; the old suspended queue remains stopped and
untouched. No cloud charge is added. The original [launch packet](RUNPOD_LAUNCH.md),
[authorization](authorization.json), [extension](duration-extension.json), live
allocation-price receipts and per-worker acceptances preserve execution history.

One calibration-only archive was lost after a failed transfer; no production
ran there. Its timing snapshot remains. The [incident](retrieval-incident.json)
and strengthened deletion gates are retained; all production/helper evidence
was verified before cleanup.

The manuscript's active figure is `draft/figures/analysis037/sgd_batch_size_compact.pdf`.
The superseded MLP-only draft copy is archived with its bytes/hash; historical
Analysis027 source data and artwork remain unchanged. The reviewed next version
and completed TODO are in `manuscript/draft/revisions/2026-11-02-feedback/`.
