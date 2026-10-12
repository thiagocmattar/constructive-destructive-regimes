# Analysis018: learning-rate-scaled kappa with gradient noise

H017 extends Analysis017's three-LR, 100-pair eta-kappa overlay to
sigma0/0.01/0.1/1/10, reusing all 15 verified Analysis014 trajectories.
No new training or optimizer runs were needed.

The [figure](figures/rosenbrock_kappa_eta_noise_overlay.pdf) has one row per sigma
and three LR curves in each panel. The left column retains the previous
gradient-only quantity; the right adds the measured normalized noise term from
the identity. Left axes share one linear Y scale; right axes use a separate
linear range for each row. All means are shown without clipping.

A separate [250-pair version](figures/rosenbrock_kappa_eta_noise_overlay_mean250.pdf)
uses the same raw pair metrics, conditions and layout with a longer trailing
mean. It has 1,750 complete means per curve, starting at update251.
The original 100-pair PDF and its plotted arrays are unchanged. See
[observation002](observations/002-mean250.md); `04_plot_mean250.py` and
`05_verify_mean250.py` generate and verify the additional version.

At sigma10 the final 100-pair gradient-only means are 1.050/1.072/1.095, while
the means including noise epsilon are 0.0186/0.00426/0.00270 at decreasing LRs.
The explicit noise contribution accounts for much of the left-column level.
The corrected quantity still differs from actual directional curvature because
these trajectories use AdamW rather than SGD. This is one paired realization.

See the [method](METHOD.md) and [observation](observations/001-noise-overlay.md)
for definitions, source identities, coverage, caption and interpretation limits.
All 29,985 pairs, 28,500 complete means per metric, and 30 plotted curves pass
independent checks; both PDFs pass visual review. Original figures remain
unchanged. Scripts `00_compute.py` through `03_record.py` regenerate the analysis;
`results.json`, `verification.json`, `figure_manifest.json`,
`figure_verification.json`, `plot_data.json` and `inventory.json` record provenance.

No cloud resources or billing. The 2.66 MB of new derived histories are retained
locally and ignored by Git; existing immutable source histories are reused.
