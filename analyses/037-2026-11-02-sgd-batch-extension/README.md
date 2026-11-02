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

**Deferred by the user:** "Keep this as a TODO, 8 hours is too long."
Preparation and bounded local calibration are retained. Spark was busy and was
not used. No production training, scientific result, cloud launch, paid storage
or running task process remains. No new hypothesis is allocated: this extends
H023's batch-noise comparison. Run214 remains reserved for this preparation;
it must not be reused for an unrelated experiment.

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

Pending before any later production: re-confirm scope/resource/budget, exercise
Transformer sampled forks and full-batch replay, qualify the complete worker
and retrieval/reducer, and calibrate the selected runtime. The prepared reducer
has not been run on production results, which do not exist. No figure or
manuscript claim was changed by this preparation.
