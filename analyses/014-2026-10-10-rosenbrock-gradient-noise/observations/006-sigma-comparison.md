# Observation006: increasing gradient noise, one paired realization

H013, completed 2026-10-10. The [method](../METHOD.md) records the user's
clarification: keep Rosenbrock and add Gaussian noise to its analytic gradient.
One seed1234 was requested for a first look. Sigma0/0.01/0.1/1/10 are compared
at LR1e-2/geometric midpoint/1e-3, start(0,1), 2,000 updates and AdamW decay0.1.
All nonzero conditions share the same standard-normal draws, scaled by sigma.
These are paired realizations, not independent replicates; later analytic
signals differ because earlier noise changes the trajectory.

Sources: [15-condition cohort](../cohort.json), [results](../results.json),
[verification](../verification.json), [figure manifest](../figures_manifest.json)
and [figure verification](../figure_verification.json). The cohort maps
reused Runs038-040 and new Runs041-052 to frozen configs and complete local
histories; manifests/receipts pin their SHA-256 hashes. All 2,001 exact
unpenalized objective values and 2,000 vectors per path are available.

| Sigma | Final f, LR1e-2 | Final f, midpoint | Final f, LR1e-3 | Late raw mean cosine, decreasing LRs | Late momentum mean cosine, decreasing LRs |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.000972263 | 0.000117958 | 0.0000110471 | -0.659 / -0.658 / -0.006 | 0.642 / 0.658 / 0.836 |
| 0.01 | 0.00110494 | 0.000134826 | 0.0000200108 | -0.520 / -0.370 / 0.310 | 0.661 / 0.660 / 0.852 |
| 0.1 | 0.00141499 | 0.000411592 | 0.000517220 | -0.112 / 0.154 / 0.342 | 0.610 / 0.662 / 0.938 |
| 1 | 0.00851233 | 0.0125985 | 0.0413583 | 0.212 / 0.144 / 0.084 | 0.717 / 0.805 / 0.906 |
| 10 | 0.117386 | 0.135555 | 0.148586 | 0.031 / -0.010 / -0.034 | 0.822 / 0.811 / 0.813 |

The final objective increases with sigma at every LR in this realization.
The last-1,000-update median objective also increases with sigma at each LR.
Best terminal LR changes from 1e-3 (sigma0/0.01), to midpoint (sigma0.1),
to 1e-2 (sigma1/10). This is a finite-budget outcome of one chosen sequence;
it does not establish an ensemble noise law or universal LR recommendation.
Every condition fails f<=1e-8 jointly with true post-gradient norm<=1e-4,
including both a first threshold hit and sustained final100 criterion.

Raw full-prefix B/A decreases at each LR as sigma grows: at sigma0 it is
0.822/0.952/0.967, and at sigma10 it is 0.223/0.502/0.790. Momentum remains
more persistent, with sigma10 B/A 0.558/0.797/0.919. These ratios accumulate
the whole history, including large early gradients; they are not local measures
of the late noisy neighborhood. Adjacent raw cosine responds nonmonotonically:
small noise can disrupt the deterministic sign alternation before larger noise
brings its late average near zero. Momentum averages remain positive, even
at sigma10 where objective values are much higher. Neither sign nor magnitude
alone demonstrates stability or objective decrease.

The identities B_g^2=E_g+I_g and
B_g^2=||H||^2+||X||^2+2<H,X> hold at every prefix up to FP64 roundoff.
For sigma10, final noise B_xi^2=407,484.553 and E_xi=404,004.892, versus their
common ensemble expectation400,000. The realized prefix cross2<H,X> is
-9.977e6/-3.118e7/-9.696e7 in decreasing LR order. It is not zero, and depends
on LR although noise draws are matched. At sigma0.01 the corresponding cross
terms are -9,363.23/-30,093.91/-93,537.17. These values describe this adapted
path; no expected sign or limiting cross term is inferred from one seed.
E[noise B^2]=E[noise energy]=2*t*sigma^2 is an ensemble statement. The energy
and temporal interaction decompositions retain same-step and cross-time terms
separately; they are verified even though the figure displays the prefix split.

Caption: **Rosenbrock loss and vector geometry as Gaussian gradient-estimation
noise increases.** The [sigma comparison PDF](../figures/sigma_comparison.pdf)
shows (a) final exact objective; (b) median exact objective over updates1001-2000;
(c,d) raw noisy-gradient and stored-momentum B/A over the full2,000-vector
prefix; (e,f) arithmetic means of unsmoothed adjacent cosine over
updates1001-2000. These are means, not medians or averages of the plotted
rolling curves. Loss axes use log Y, ratios/cosines linear Y. Sigma X uses
symmetric log with a linear region through sigma0.01, so the deterministic
sigma0 control is visible. Each marker is one path; lines connect sampled
noise levels and are not a fitted response. Colors and line styles identify
the same three LRs in every figure. No error bars or replicate bands.

Reduction: A=sum vector norms; B=norm vector prefix sum. First adjacent pair,
zero-vector cosine and A=0 ratio are undefined without added epsilon.
All observed raw/momentum vectors are nonzero. Late means weight 1,000 pairs
equally; late loss medians use 1,000 post-update values. No smoothing,
thinning, clipping, extrapolation or seed averaging in the sweep figure.
The [five per-sigma observations](INDEX.md) provide the ten navigation/identity
captions, including the 100-update cosine mean used only in navigation PDFs.

Generating script: [02_plot.py](../02_plot.py), using the shared navigation
plotter and exact noise decomposition. [03_record.py](../03_record.py) writes
all reductions and checks every rolling mean, prefix energy, PDF and source.
All 24,000 new optimizer updates also pass independent PyTorch comparison;
all old controls pass exact zero-noise replay without modifying their evidence.
Interpretation is limited to this function, initialization, optimizer,
absolute isotropic noise scale, horizon and single paired seed. No observed
minibatch noise model, causal alignment mechanism, generalization or LLM claim.
