# Observation001: gradient noise sigma 0

H013: compare the same three LRs at sigma=0, using the
[pre-execution method](../METHOD.md). Raw g=grad f(theta_previous)+xi;
xi is N(0,sigma^2 I_2), noise seed1234. Sigma0 reuses deterministic controls.
Sources: [cohort](../cohort.json), full run histories and receipts,
[results](../results.json), [verification](../verification.json), and
[figure manifest](../figures_manifest.json), pinning sources/scripts/PDF hashes.
Reduction: all2,001 exact objective values and2,000 raw/momentum vectors for
each LR. A=sum vector norms, B=norm vector sum, ratio B/A. First cosine and
zero-norm pairs are undefined; A=0 ratios undefined; no epsilon. All observed
raw/momentum vectors are nonzero. Table summaries are unsmoothed, equal-pair
weighted, with late statistics using the last1,000 updates. One realization,
no uncertainty bands or sampling-error estimate. LR also multiplies decay.

| LR | Final f | Late median f | Raw B/A | Momentum B/A | Late raw mean cosine | Late momentum mean cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-2 | 0.00097226306 | 0.000991059061 | 0.822310 | 0.920950 | -0.658802 | 0.641664 |
| midpoint | 0.000117958119 | 0.000111891083 | 0.952012 | 0.961221 | -0.657991 | 0.658197 |
| 1e-3 | 1.10471357e-05 | 2.93981862e-05 | 0.966548 | 0.967110 | -0.006125 | 0.835946 |

Caption 1: **Rosenbrock navigation with sigma=0 additive Gaussian
gradient noise.** [Navigation/metrics PDF](../figures/rosenbrock_sigma_0.pdf)
shows(a) full contours and trajectories, common start(0,1), optimum(1,1), final
points; (b) exact unpenalized objective; (c,d)100-update trailing raw noisy-gradient
and momentum cosine means, complete finite windows only, first at update101;
(e) raw/momentum norms; (f,g) cumulative A and B; (h,i) B/A. Norm/A/B axes use
linear Y, loss log Y; common full landscape domain, no zoom/thinning. The
analytic signal and added noise are distinct retained streams. First y-turn
markers are not interpreted under the stochastic paths.

Caption 2: **Pathwise squared-sum identities and noise realization, sigma=0.**
[Identity PDF](../figures/identities_sigma_0.pdf) uses one LR per column.
Top row: B_g^2, energy E_g=sum||g||^2, exact all-pair I_g, with B_g^2=E_g+I_g.
Middle row: signal squared sum||H||^2, noise squared sum||X||^2, cross2<H,X>,
whose sum equals B_g^2. Bottom row: realized noise B_xi^2 and E_xi alongside
the dotted ensemble expectation2*t*sigma^2. Show every prefix with symmetric
log Y(linear within[-1,1]), common limits within each row. The expectation is
not a pathwise equality; cross terms are retained along the adapted path.
Sigma0's noise/cross terms are exactly zero. No sampled histories are averaged
across seeds, since there is one seed. Each identity holds up to reported FP64
roundoff; stochastic estimation changes its components, not its algebra.

Generating script: [02_plot.py](../02_plot.py) with shared plot/noise geometry;
[03_record.py](../03_record.py) writes reductions and validates every rolling
mean, energy prefix, source identity, PDF page/text and hash. See
[figure verification](../figure_verification.json) for rendered review.
Interpretation limits: descriptive finite-budget outcomes, no causal alignment
mechanism, formal stability/generalization, ensemble monotonicity or LLM transfer.
