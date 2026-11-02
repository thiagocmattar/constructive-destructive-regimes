# Compact batch-size comparison, 2026-11-02

`figures/sgd_batch_size_compact.pdf` implements the author's requested 2x2
view: initial and 2,000-GD-update starting states; sampled-gradient mean
cosine and 200-update displacement retention. The manuscript copy is under
`manuscript/draft/figures/analysis027/`. The omitted intermediate state and
full-gradient curves remain in the historical figures and original evidence.

Sources: passed `verification.json`, `results.json`, `reduction.json` and
`scientific_summary.json`, all hash checked. The revision verifier also
reconstructs the 60 plotted initialization-level values from the underlying
repetition-level results. Each stochastic condition has eight repetitions;
full batch has one. Coverage is the complete 5,000-image MLP objective,
three initializations, five batch sizes, two selected initial states and
200 updates per continuation. LR stays 0.02. Equal updates are unequal exposure.

Reduction: mean over all 199 defined cosine pairs; complete-prefix B/A over
200 updates. Average repetitions within initialization, then equal-weight
the three initialization means. One line per panel is this mean; shading
is the initialization range, not a confidence interval. No smoothing.
Axes use a logarithmic batch coordinate and explicitly label 5,000 as full.

Observation: reducing batch size from full GD weakens initial positive
alignment and lowers retention; from the trained state it attenuates negative
alignment while retention stays near 0.03. Thus noisy directional variation
and systematic reversal can both produce low retention. The smallest batch
increases trained-state endpoint loss while full GD decreases it. Cosine
attenuation alone is not an optimization or generalization benefit. This is
one objective and LR, not an isotropic-noise or universal batch-size law.

Generator: `manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py`.
Caption: `manuscript/draft/sections/05-sgd-noise-experiments.tex`.
Source/figure hashes and completed review are in the revision's
`figure_manifest.json`, `verification.json` and `review.json`.
No new sampling, training or cloud execution.
