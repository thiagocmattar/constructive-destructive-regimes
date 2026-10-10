# Analysis014: Gaussian estimation noise in Rosenbrock gradients

Completed H013 on 2026-10-10: one paired noise seed, five noise levels and the
same three learning rates as Analysis013. The clarified stochastic estimator is
**g_t = grad f(theta_(t-1)) + xi_t**, with xi_t ~ N(0, sigma^2 I_2).
The objective remains Rosenbrock, f(x,y)=(1-x)^2+100(y-x^2)^2.
Sigma is a coordinate standard deviation in gradient units, not a percentage.

Start (0,1), 2,000 updates, AdamW betas (0.9,0.95), epsilon 1e-8, decoupled
weight decay 0.1, constant LR, zero moments, CPU float64 and no clipping.
LRs are 1e-2, the geometric midpoint 0.0031622776601683794, and 1e-3.
The same PCG64 seed1234 standard-normal sequence is scaled across all nonzero
sigmas and reused across LRs. Sigma0 reuses Runs038-040 without rewriting them;
Runs041-052 contain the 12 new trajectories. Each figure has three paths.

| Sigma | Navigation and eight metrics | Squared-sum identities | Final f: LR 1e-2 / midpoint / 1e-3 |
| ---: | --- | --- | --- |
| 0 | [PDF](figures/rosenbrock_sigma_0.pdf) | [PDF](figures/identities_sigma_0.pdf) | 0.000972263 / 0.000117958 / 0.0000110471 |
| 0.01 | [PDF](figures/rosenbrock_sigma_1e-2.pdf) | [PDF](figures/identities_sigma_1e-2.pdf) | 0.00110494 / 0.000134826 / 0.0000200108 |
| 0.1 | [PDF](figures/rosenbrock_sigma_1e-1.pdf) | [PDF](figures/identities_sigma_1e-1.pdf) | 0.00141499 / 0.000411592 / 0.000517220 |
| 1 | [PDF](figures/rosenbrock_sigma_1.pdf) | [PDF](figures/identities_sigma_1.pdf) | 0.00851233 / 0.0125985 / 0.0413583 |
| 10 | [PDF](figures/rosenbrock_sigma_10.pdf) | [PDF](figures/identities_sigma_10.pdf) | 0.117386 / 0.135555 / 0.148586 |

The [sigma comparison PDF](figures/sigma_comparison.pdf) puts final/late loss,
raw/momentum B/A and late adjacent cosine on the same noise-level axis.
Full captions, sources, reductions and limits are in the
[seven observations](observations/INDEX.md), especially the
[cross-sigma interpretation](observations/006-sigma-comparison.md).
The final [5x3 norm grid](figures/gradient_momentum_norms_grid.pdf) has one LR
per column, one sigma per row and both raw/momentum norms at every update,
with shared log Y axes for comparison.

In this realization, final objective increases with sigma at each LR, and raw
full-prefix B/A decreases. The lowest terminal loss moves from LR1e-3 at
sigma0/0.01, to the midpoint at sigma0.1, to LR1e-2 at sigma1/10. At sigma10,
late mean raw cosine is 0.031/-0.010/-0.034, whereas momentum is
0.822/0.811/0.813. Persistent momentum therefore coexists with higher loss.
None of the 15 paths meets the joint true-objective/gradient criterion, even
once. One matched realization provides no ensemble monotonicity or uncertainty
estimate, asymptotic convergence claim, causal alignment benefit or LLM result.

The exact identities still hold pathwise: B^2=E+I, where E=sum||g_t||^2 and
I=2 sum_(r<s)<g_r,g_s>. The noisy prefix additionally satisfies
B_g^2=||H||^2+||X||^2+2<H,X>, H=sum grad f(theta_previous), X=sum xi.
Noise changes these components; centered draws do not remove realized cross
terms, especially on a path adapted to previous noise. Noise-only B_xi^2 and
energy have ensemble expectation 2*t*sigma^2, not pathwise equality.
A=sum||g_t||; A^2-B^2 is a triangle gap, distinct from I.

The navigation figures retain the large full contour/trajectory panel and the
requested two-column grid. Panels(c,d) use complete 100-update trailing cosine
means, first displayed at update101. Norm/A/B Y axes are linear; loss Y is log.
Raw means the noisy estimator used by AdamW; momentum means the stored first
moment, before bias correction, adaptive scaling and decay. All figures share
one landscape domain; all updates and prefixes are retained. Stochastic first
local y reversals are not treated as the prior deterministic valley-entry event.

Numerical code and executed inputs are frozen. Every draw, analytic signal,
noisy gradient, moment, adaptive direction, displacement and geometry prefix is
retained locally. The 12 new NPZ histories occupy **18,632,620 bytes (18.63 MB)**;
reported numerical execution/saving sums to **0.806 s**, excluding verification
and plotting. All work used local CPU; no cloud resources or billing.

Verification covers exact zero-noise baseline reproduction, same draws and
correct placement, analytic objective/gradient against SciPy, full replay,
independent PyTorch AdamW at all 24,000 new updates, all six vector streams,
prefix/temporal/noise/momentum identities, decay displacement, RNG state and
artifact hashes. Independent optimizer position error is at most 3.33e-16.
Every rolling mean and energy prefix is independently reduced. All 12 PDF pages
pass source/hash/text/bounds checks and rendered layout review.

Records: [method before execution](METHOD.md), [cohort](cohort.json),
[prelaunch mechanics](prelaunch_verification.json), [numerical verification](verification.json),
[results](results.json), [figure manifest](figures_manifest.json),
[figure verification](figure_verification.json).

Reproduction: [00_prepare.py](00_prepare.py) allocates inputs only once;
[00_execute.py](00_execute.py) refuses to overwrite existing attempts.
[01_verify.py](01_verify.py) replays in memory and verifies retained evidence.
[02_plot.py](02_plot.py) regenerates figures, and [03_record.py](03_record.py)
reduces records, checks PDFs and creates scratch renders. The shared
[stochastic engine](../../src/controlled/stochastic_rosenbrock.py) and
[noise identities](../../src/controlled/noise_geometry.py) are hash-pinned in
each executed manifest. Changed scientific inputs require new run IDs.
