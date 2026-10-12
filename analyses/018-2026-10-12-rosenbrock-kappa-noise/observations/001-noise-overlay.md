# 001: Noise changes the eta-kappa identity overlay

Question: how does the previous eta-scaled gradient-identity overlay change
when the stochastic-gradient estimator has increasing additive Gaussian noise?
H017 is exploratory; no directional prediction was specified.

## Sources, coverage and reduction

Use all 15 immutable histories from [Analysis014](../../014-2026-10-10-rosenbrock-gradient-noise/README.md),
Runs038-052, as enumerated in [cohort.json](../cohort.json). Each source's
history, config and verification receipt SHA-256 is in [results.json](../results.json).
Sigma0 data agree with the previous [Analysis017 overlay](../../017-2026-10-12-rosenbrock-kappa-identity/observations/002-eta-kappa-overlay.md).
The formula uses Analysis017's frozen restatement snapshot; its hash is recorded.

Match objective, start(0,1), 2,000 updates, AdamW betas(0.9,0.95), numerical
epsilon1e-8, weight decay0.1 and float64. Cross five sigmas0/0.01/0.1/1/10
with constant LRs0.01/0.0031622776601683794/0.001. One PCG64 seed1234 provides
the same standardized noise vectors across all nonzero sigmas/LRs. Raw g is the
noisy gradient supplied to the optimizer, not the exact objective gradient.

For each adjacent pair compute q = 1-rho*norm(g_current)/norm(g_previous),
epsilon = dot(g_previous, xi_current-xi_previous)/norm(g_previous)^2,
and eta*kappa_proxy = q+epsilon. Epsilon is the realized noise contribution,
not AdamW's numerical epsilon. The left column omits epsilon to repeat the old
observable; the right includes it. Sigma0 makes both columns identical.
Complete 100-pair arithmetic means are taken after computing the per-pair ratios.

Each curve has 1,999 valid pairs at t2..2000 and 1,900 full means at t101..2000.
All 29,985 pairs and 28,500 means per metric are checked independently.
No clipping, denominator floors, outlier removal, partial windows, thinning or
seed averaging. The first update is undefined; zero adjacent norms would be
undefined (none occur). Exact plotted means are in [plot_data.json](../plot_data.json).

## Results

The deterministic plateau around1.75-1.79 becomes less persistent as noise
increases. At sigma0.01 and0.1, values fluctuate and the three LRs no longer
share the same late level. At sigma1 and10, the left-column values spend much
of the late trajectory around1, while the right-column scale becomes smaller.
These changes are descriptive, not monotone at every time or learning rate.

Final 100-pair means, in decreasing LR order:

| Sigma | Gradient-only LR0.01 | Gradient-only midpoint | Gradient-only LR0.001 | Including epsilon LR0.01 | Including epsilon midpoint | Including epsilon LR0.001 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.790182 | 1.754971 | 1.747944 | 1.790182 | 1.754971 | 1.747944 |
| 0.01 | 1.572252 | 0.861300 | 1.548070 | 1.599745 | 0.844882 | 1.169615 |
| 0.1 | 1.352702 | 0.806380 | 0.837741 | 1.040299 | 0.622477 | 0.247784 |
| 1 | 1.021011 | 0.951529 | 0.869116 | 0.245211 | 0.071466 | 0.025279 |
| 10 | 1.050159 | 1.071880 | 1.095094 | 0.018634 | 0.004262 | 0.002698 |

At sigma10, final100 epsilon means are -1.031525/-1.067618/-1.092396.
Adding this measured term cancels almost all of the uncorrected level near1.
This is exact realized algebra and demonstrates why treating epsilon as zero
would confound this observable's noise and curvature-related contributions.
Even small absolute noise can affect the ratio when signal norms are small.

The corrected value is not actual directional curvature under AdamW.
At sigma10, final100 eta times exact Rayleigh curvature is
2.620993/0.700680/0.215342, versus corrected values
0.018634/0.004262/0.002698. AdamW mismatch contributes
-2.602445/-0.696426/-0.212646 and nonlinear remainder
0.000087/0.000008/0.000001. Thus correction removes the explicit noise
difference, while the optimizer mismatch remains dominant in this late window.
The exact decomposition and synthetic noisy quadratic SGD are independently
verified; see [METHOD.md](../METHOD.md).

## Caption and legend

**Learning-rate-scaled gradient-identity diagnostics under increasing Rosenbrock
gradient noise.** Rows show coordinate noise standard deviations sigma0,0.01,
0.1,1,10. Blue solid, purple short dashed, and orange long dashed lines show
constant LRs0.01,0.0031622776601683794,0.001. Left panels display the100-pair
trailing mean of q=1-rho*norm(g_t)/norm(g_previous), matching the previous
normalized observable. Right panels display the100-pair mean of
eta*kappa_proxy=q+epsilon using the measured normalized noise contribution.
Left Y limits are common; right Y ranges differ by row, explicitly indicated,
to resolve smaller corrected values. All axes are linear. The first full mean
is update101; all1,900 means per curve are shown. Initial point(0,1),2000updates,
AdamW weight decay0.1, one paired seed1234. Curves are SGD-derived proxies on
AdamW and are not a curvature or stability estimator validation.

## Uncertainty, caveats and provenance

One paired seed supplies no ensemble intervals or universal noise law.
Overlapping windows are dependent. Relative noise changes along each path;
near-zero preceding norms can amplify raw pair ratios, all of which are retained.
Noise changes both gradient estimates and subsequent positions. Correcting
epsilon cannot undo that path change, AdamW scaling/momentum or weight decay.
No asymptotic convergence, stability threshold or transfer to model training
is established. Deterministic valley-turn annotations are not reused for noisy paths.

Generating scripts: [00_compute.py](../00_compute.py), [01_verify.py](../01_verify.py),
[02_plot.py](../02_plot.py), [03_record.py](../03_record.py).
Output: [PDF](../figures/rosenbrock_kappa_eta_noise_overlay.pdf).
Numeric/source/plot checks: [verification.json](../verification.json),
[figure_manifest.json](../figure_manifest.json), [figure_verification.json](../figure_verification.json).
All30 plotted curves and57,000 finite points match the derived arrays exactly;
the complete PDF render was visually inspected. Original Analysis017 PDFs are
unchanged. New derived histories2,655,983bytes are retained outside Git.
No new optimizer trajectories, cloud resources or billing.
