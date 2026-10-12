# Observation001: gradient identity along AdamW Rosenbrock trajectories

H016 asks whether the requested identity has a shape similar to lambda_max.
Sources: immutable Run055/056/057 histories, configs and verification receipts;
[results](../results.json) pins their SHA-256/size and all derived array hashes.
The supplied formula source is manuscript/2026-10-11-restatement/problem.md,
also hashed. Full diagnostic definitions/indexing are in [METHOD](../METHOD.md).
Reduction: ../00_compute.py and ../kappa.py; verification: ../01_verify.py;
plotting: ../02_plot.py and ../plot_full.py; figure acceptance: ../03_record.py.

| LR | Raw Pearson vs lambda_max | 100-pair Pearson | Late proxy mean | Late exact directional mean | Late lambda_max mean | Late eta-proxy mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.406933 | 0.942175 | 179.018242 | 903.357130 | 953.684190 | 1.790182 |
| 0.003162278 | 0.507985 | 0.905473 | 554.970525 | 933.010424 | 985.285718 | 1.754971 |
| 0.001 | 0.568844 | 0.753220 | 1747.943867 | 927.394838 | 996.310036 | 1.747944 |

Each LR has1,999 defined unsmoothed pairs t=2..2000 and1,900 complete100-pair
means t=101..2000. Late arithmetic means use pairs1901..2000, whose Hessian
positions are theta[1899..1998]. Initial pair is undefined; no epsilon floor,
clamping, thinning or curve normalization. The identity is computed before
smoothing. H excludes decoupled decay and uses the preceding Taylor point.

Smoothed correlations are higher than raw correlations, but amplitudes differ
substantially. Proxy values outside the local Hessian spectrum occur at
245/326/728 pairs. Exact directional curvature stays within the spectrum.
Late proxy-minus-direction means are explained by update-mismatch corrections
-724.340/-378.043/+820.549; late nonlinear-remainder means are approximately
0.000830/0.002783/-0.0000149. The exact algebraic decomposition closes within
2.4e-12 at every pair. This identifies terms of the discrepancy, not a causal
benefit. Under plain SGD on a quadratic, the expression recovers the Rayleigh
quotient; even that quotient need not be lambda_max.

Caption - preserved [six-panel comparison](../figures/rosenbrock_kappa_comparison.pdf):
**Gradient-identity kappa versus exact Rosenbrock curvature.** Columns are
LR0.01/midpoint/0.001, blue/purple/orange. Top: all unsmoothed proxy pairs,
largest eigenvalue (dark solid), and exact directional curvature (green dashed).
Bottom:100-pair trailing arithmetic means of each, common linear Y limits
within each row. No clipping or rescaling. LR-coloured vertical references
are original first-y-turn positions81/228/655, not estimator thresholds.

Caption - [full diagnostics](../figures/rosenbrock_kappa_diagnostics.pdf):
**Analysis016 geometry with the requested identity.** Complete trajectory
on original contours, square start, star optimum, open final points and diamond
y-turns; loss;100-step raw/momentum cosine means; vector norms; raw/momentum
A and B and B/A; largest/smallest Hessian eigenvalues;100-pair proxy means with
faint originals; exact directional curvature. All temporal axes linear except
loss logarithmic. Original Hessian panels use theta[t]; pair-curvature panels
use theta[t-2]. Every source point/vector remains visible, without zoom.

Coverage is complete, uncertainty is not estimated: one deterministic function,
start, optimizer and horizon. Serial correlations are descriptive and sensitive
to averaging. The SGD derivation is applied as a proxy to AdamW, whose moments,
preconditioning and decay change its step. No valid curvature estimator,
universal stability threshold, asymptotic convergence or LLM transfer is shown.
Both PDFs pass actual-series/source/text-bound checks and final layout inspection
as recorded in [figure verification](../figure_verification.json).
