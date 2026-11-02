# Analysis037: three-case batch-size and starting-state comparison

H023; Runs214/216. User requested measurements for the confirmed existing cases:
CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer. Reuse Analysis027
MLP results. Add ResNet/Transformer continuations at initialization and after
2000 GD updates, then replace the manuscript figure with two rows and three
columns, with two starting-state curves in each panel.

The [measurement contract](../../manuscript/draft/revisions/2026-11-02-feedback/SGD_EXTENSION.md)
defines comparisons, coverage, aggregation, interpretation limits and retention.
Run214's config makes the new-reference policy explicit: all seeds use fresh
2000-update GD trajectories on the selected per-case runtime, from pinned
initial tensors. This adds12000 reference updates to69600 continuation updates.
It avoids combining historical seed0 numerical trajectories with new seeds.
Run216 separately freezes Transformer physical accumulation512, selected by
pre-production complete-gradient timing/precision checks; ResNet uses Run214.

The user resumed execution on Runpod after previously deferring the eight-hour
proposal, approved US$40/four hours, and then explicitly extended the shared
wall limit to six hours while keeping US$40 total. First allocation:
2026-11-02 13:34:56 UTC; hard deadline19:34:56 UTC /16:34:56 São Paulo.
All six task/seed blocks now run in parallel on qualified workers. Calibration,
capacity retries, storage and verified retrieval count toward the same ceiling.
The [original launch packet](RUNPOD_LAUNCH.md), [authorization](authorization.json)
and [duration extension](duration-extension.json) preserve the decision record.
No new hypothesis is allocated: this extends H023's batch-noise comparison.

Seven CPU checks pass: direct all-pair geometry, zero/cancelling gradients,
unbiased variable-token sampling, fixed ghost groups, paired batch sampling,
RNG/geometry recovery, and actual versus nominal displacement. Local GPU checks
cover all batch sizes; the ResNet reference stopped/resumed and completed a
sampled fork, and the Transformer reference completed three updates. These are
mechanics checks, not evidence for the paper.

The [preparation receipt](preparation_receipt.json) preserves source hashes,
timing/memory and completed checks. ResNet full updates take about7.56 seconds;
Transformer full updates about16.13 seconds on the owned5070Ti laptop. A rough
linear projection plus25% allowance yields about60-75 hours for the full design;
short timing samples do not measure sustained thermal throughput. The proposed
Runpod8-hour limit was a maximum execution envelope, not a measured cloud ETC.
The read-only [catalog snapshot](live_catalog.json) incurred no spending.

The [worker qualification](worker_qualification.json) now covers both complete
short queues, sampled forks, independently executed full-batch replay, saved
sampling-index replay, CPU vector checks, copied-artifact hash verification and
deliberate corruption rejection. All 78 inventory entries pass. Transformer
replay is bitwise identical. ResNet's maximum absolute parameter difference is
2.97e-8, relative total L2 error 1.07e-9; bitwise replay failed and the explicit
FP32 tolerances are recorded. This does not qualify long-trajectory replay.

Pending: complete the full cohort, independently reduce all arms, regenerate
the figure and integrate the manuscript. No new scientific result or manuscript
claim is supplied by mechanics checks. The original preparation receipt remains
unchanged; the current input manifest has identical scientific values but adds
all run-script hashes and uses LF. The short workers' CRLF manifest is retained
under ignored scratch and identified separately in the qualification receipt.

## Active execution and retention

ResNet seed0/2 use L40S (US$1.09/GPU-hour), seed1 uses RTX4090 (US$0.89);
Transformer uses RTX5090 (US$1.19) for all three blocks. These rates come from
each allocation's live catalog receipt. Full-gradient calibration is about
2.74--2.84 seconds on L40S,2.6 seconds on4090 and4.0--4.1 seconds on5090 for
their respective workloads. Short calibrations are qualified estimates, not
guaranteed sustained timings. Slower/unreachable calibration resources were
released; their cost remains included. Extra-worker requests continue across
GPU classes when useful for the remaining critical path.

All production inputs stay frozen. Detached workers save recovery every50
updates, exact sampling indices, scalar histories, endpoints and audit vectors.
The independent guard stops science before the hard deadline or spending cap;
immutable completed forks stream to local storage during computation.
`14_accept_workers.py` verifies the complete raw inventory, numerical reduction,
sampling replay and displacement before allowing that worker's deletion.
Unrelated account Pods and network volumes are preserved.

After the user identified idle Spark capacity, a separate GB10 task directory
qualified the unchanged Run214 inputs under PyTorch2.11/CUDA13. Initial complete
gradient relative L2 difference versus L40S is1.20e-7 (cosine1-6.9e-15);
three-update parameter relative L2 difference is1.59e-9. Full updates take13.3s,
so Spark handles only ResNet seed2's32 small-group forks (1/5 groups at both
states), while its cloud worker retains the reference and larger batches.
This is execution parallelism, with explicit additional runtime variation.
An extra initial200-update full-GD control measures longer runtime drift.
The [qualification](spark-qualification.json), [launch](spark-launch.json) and
handoff receipts preserve its scope; raw inputs remain unchanged.
Each imported fork is byte/hash checked and its sampling/geometry reduced
before atomic placement ahead of the cloud queue. Completed cloud-worker
acceptance still checks every imported fork independently. No cloud charge
is added, and the old suspended Spark queue is untouched.

Transformer seed0 and ResNet seed2 initial full-GD controls use their exactly
captured first200 reference updates, saving400 executed updates without losing
either trajectory.
The retained checkpoint, loss evaluation and audit limitation are explicit in
[the derivation plan](initial-gd-control-reuse-plan.json). Other full controls
execute independently. Coverage remains348 forks plus six references.

One calibration-only Pod was mistakenly deleted after an archive transfer
failed. Its timing snapshot remains, but its full mechanics archive is lost;
it had no production measurements. The [incident record](retrieval-incident.json)
documents this and the deletion gates added afterward. This limitation must not
be hidden by later successful production retrievals.
