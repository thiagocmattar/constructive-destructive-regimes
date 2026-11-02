# Three-case batch-size manuscript figure

Status: passed complete-cohort, numerical and visual review.

Figure: [sgd_batch_size_compact.pdf](../figures/sgd_batch_size_compact.pdf).
The active manuscript copy is `manuscript/draft/figures/analysis037/`.
This replaces the earlier MLP-only layout with the user-confirmed three cases,
two metric rows and two starting-state curves per panel.

Sources: [results.json](../results.json),
[cohort_acceptance.json](../cohort_acceptance.json), and unchanged Analysis027
MLP `results.json`/`reduction.json` with their passed verification receipts.
Runs214/216 retain the new raw scalar histories, sampling indices, checkpoints,
audit vectors, identities and exact byte/SHA inventories under ignored artifacts.
The revision's `figure_manifest.json` records source, array and output hashes;
its independent verifier reconstructs all 168 initialization-block scalar values
from the underlying repetition-level results.

Coverage: three initializations per case, starts 0/2000, and 200 updates per fork;
eight repetitions per stochastic condition and one full-GD control per block.
The figure uses 198 existing MLP,150 new ResNet and 198 new Transformer
continuations (546 total), with no missing cases filled by interpolation.
MLP's complete objective has 5,000 images; ResNet has 50,000 images in 50 fixed
normalization groups; Transformer has 59,680 sequences and 2,088,600 supervised
tokens. Rates are 0.02/0.1/0.2, respectively. Data/initial tensors and scientific
inputs are hash pinned. The additional Spark runtime bridge is not a replicate.

Reduction: mean cosine over 199 adjacent raw sampled-gradient pairs; nominal
gradient-sum displacement retention over all 200 updates. Actual FP32 increments
and endpoint displacement are audited separately. Average repetitions within
initialization, then weight the three initialization means equally. Shading is
their range, not a confidence interval. No smoothing; log batch axes; hollow
rightmost markers distinguish full GD. Lines only connect measured batch sizes.

Observation: the smallest batches weaken trained-state reversal across all
three cases, while every trained-state retention mean remains below 0.05.
Initial-state MLP sampled paths can instead change alignment sign relative
to full GD. Transformer batch 8 has mean cosine near-0.26 from both starting
states, but retention changes from 0.110 to 0.042 and mean complete-training-loss
change from-3.505 to+0.064. The loss-change sign holds for each block mean.
Thus local adjacent alignment does not determine cumulative movement or loss
progress. Evolving sampled trajectories cannot be reduced to independent noise
added at two fixed parameter states. See the [report](../REPORT.md) for the
complete cohort interpretation and limits.

Uncertainty/caveats: only three initialization blocks and selected rates;
equal updates are unequal exposure. Data, architecture and objective differ
across cases, preventing attribution to any one factor. ResNet uses L40S/4090
and seed 2 small-group forks use GB10; runtime and initialization variation are
not separated by shading; batch and runtime effects are confounded within
that seed block. The initial 200-update runtime bridge bounds only
that replay. Neither stability, covariance structure nor generalization is
measured by these two rows.

Generator: `manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py`.
Independent reduction: `02_verify_results.py`, `15_verify_cohort.py`, and the
revision's `02_build_and_verify.py`. Caption source:
`manuscript/draft/sections/05-sgd-noise-experiments.tex`.

Caption: Batch-size and starting-state comparison across three cases. Top:
mean sampled-gradient cosine over 199 adjacent pairs. Bottom: displacement
retention over the 200-update continuation. Blue circles start at initialization;
orange squares start after 2,000 GD updates. Lines average repetitions within
initialization and then weight the three blocks equally; shading is their
range, not a confidence interval. Hollow rightmost markers denote full GD.
ResNet batches contain whole fixed 1,000-image normalization groups;
Transformer batches count prepared sequences. Batch axes are logarithmic;
lines guide the eye between measured sizes.
