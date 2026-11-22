> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-three-case-sgd.md](../../../../../analyses/037-2026-11-02-sgd-batch-extension/observations/001-three-case-sgd.md).

# Three-case mean-cosine manuscript figure

Status: passed complete-cohort, numerical and visual review.

Figure: [sgd_batch_size_compact.pdf](../../../../../analyses/037-2026-11-02-sgd-batch-extension/figures/sgd_batch_size_compact.pdf).
The manuscript copy is `manuscript/draft/figures/analysis037/`. The latest author
feedback removes the displacement-ratio row: one row of three case panels now
shows only mean adjacent cosine, with initialization and 2k-state curves.
The earlier two-row view is preserved at commit 093b81c. Measurements are unchanged.

Sources: [results.json](../../../../../analyses/037-2026-11-02-sgd-batch-extension/results.json),
[cohort_acceptance.json](../../../../../analyses/037-2026-11-02-sgd-batch-extension/cohort_acceptance.json), and the unchanged Analysis027
MLP results/reduction with passed verification. Runs214/216 preserve raw scalar
histories, sampling indices, checkpoints, audit vectors and exact byte/SHA
inventories under ignored artifacts. The revision's figure manifest records
source, array and output hashes; its verifier reconstructs all 84 plotted
initialization-block scalar values from the underlying repetitions. All six
cosine curves retain exactly the values and ranges of the previous layout.

Coverage: three initializations per case, starts 0/2000, 200 updates per fork;
eight repetitions per stochastic condition and one full-GD control per block.
There are 198 reused MLP, 150 new ResNet and 198 new Transformer continuations
(546 total). MLP uses 5,000 images, ResNet 50,000 images in 50 fixed normalization
groups, and Transformer 59,680 sequences/2,088,600 supervised tokens. Rates are
0.02/0.1/0.2. Inputs are hash pinned; the extra Spark bridge is not a replicate.

Reduction: average 199 adjacent raw sampled-gradient cosines per continuation,
then repetitions within initialization, then the three block means equally.
The six curves contain 28 plotted means; shading spans the three block means,
not a confidence interval. No smoothing. Batch axes are logarithmic; hollow
rightmost markers denote full GD. Lines connect only measured batch sizes.
Retention and its actual-update audits remain in the supporting results.

Observation: smallest batches weaken trained-state reversal in all three cases.
Initial-state MLP sampled paths can have negative mean alignment while its
full-GD mean is positive. Transformer batch 8 has mean cosine near -0.26 from
both starts, yet opposite mean complete-training-loss changes. Adjacent cosine
alone does not order objective progress. The [report](../../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md) retains
retention results and the full measurement interpretation.

Limits: three initialization blocks and selected rates; equal updates are
unequal exposure. Cases differ in objective, data and architecture. ResNet uses
L40S/4090 with seed 2 small-group forks on GB10; runtime and initialization
variation are not separated, and batch/runtime effects are confounded in that
block. The measured 200-update initial runtime bridge does not validate all
long replays. This figure establishes no stability or generalization benefit.

Generator: `manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py`.
Independent checks: the owning analysis's `02_verify_results.py` and
`15_verify_cohort.py`, plus the revision's `02_build_and_verify.py`.
Caption: `manuscript/draft/sections/05-sgd-noise-experiments.tex`.

Caption: Mean sampled-gradient cosine over 199 adjacent pairs during 200-update
continuations in three model/dataset cases. Blue circles start at initialization;
orange squares start after 2,000 GD updates. Lines average repetitions within
initialization, then weight the three blocks equally; shading is their range,
not a confidence interval. Hollow rightmost markers denote full GD. ResNet
batches contain whole fixed 1,000-image normalization groups; Transformer batches
count prepared sequences. Batch axes are logarithmic; lines guide the eye.
