# Analysis037: three-case batch-size and starting-state comparison

H023; Run214. User requested measurements for the confirmed existing cases:
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

The user resumed execution on Runpod after previously deferring the eight-hour
proposal. The [launch packet](RUNPOD_LAUNCH.md) is ready; a new aggregate cost/time
envelope is pending. No production training or paid allocation has started.
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

Pending: spending envelope, selected cloud runtime/calibration and actual SSH/SCP
transfer qualification; then the full cohort, independent aggregate reduction,
figure and manuscript integration. No new scientific result or manuscript claim
is supplied by these mechanics checks. The original preparation receipt remains
unchanged; the current input manifest has identical scientific values but adds
all run-script hashes and uses LF. The short workers' CRLF manifest is retained
under ignored scratch and identified separately in the qualification receipt.
