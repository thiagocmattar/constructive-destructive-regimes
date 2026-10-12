# Observation002: separate learning-rate-scaled kappa overlay

The user explicitly requests another PDF, preserving the original comparison.
Plot eta*kappa_proxy in a single panel for all three LRs. Since eta is constant,
this cancels the explicit1/eta factor and equals1-rho*||g[t]||/||g[t-1]|| with
noise epsilon zero. This is multiplication by LR, not division or rescaling to
a fitted maximum. Compute this dimensionless value per unsmoothed pair, then
take a complete100-pair trailing arithmetic mean. First mean is at101;
all1,900 means per LR remain. The original comparison PDF's SHA-256 is checked
before and after generation and recorded unchanged in the new figure manifest.

Sources: [results](../results.json), immutable Run055-057 histories and derived
kappa arrays. Exact overlay values, null prefixes and source hashes are in
[eta_overlay_data.json](../eta_overlay_data.json). Generator:
[04_plot_eta_overlay.py](../04_plot_eta_overlay.py); independent direct-vector
identity/mean checks run in that script; plotted-array/PDF checks and review
are in ../03_record.py and [figure verification](../figure_verification.json).

Caption - [separate overlay](../figures/rosenbrock_kappa_eta_overlay.pdf):
**Learning-rate-scaled gradient-identity kappa along matched Rosenbrock AdamW
trajectories.** Blue solid: LR0.01; purple short dashed: midpoint0.003162277660;
orange long dashed: LR0.001. Single linear Y axis, all pair times,100-pair means,
no thinning, clipping, fitted normalization or uncertainty bands. Vertical
dashed references retain original y-turn indices81/228/655. Start(0,1),
2,000 updates, betas(0.9,0.95), decoupled decay0.1, CPU float64. Epsilon is
the normalized noise term (zero), separate from optimizer epsilon1e-8.

Final100-pair means are1.790182/1.754971/1.747944 at decreasing LRs. The late
dimensionless levels are similar, while the rise occurs later at smaller LR.
This cancels a formula factor; it does not normalize Hessian eigenvalues or
correct the mismatch between AdamW updates and the SGD curvature derivation.
One deterministic setup provides no ensemble uncertainty, universal plateau,
stability boundary or causal statement. Latest PDF was rendered with PyMuPDF
and inspected after layout adjustment; text, legends and footnotes are readable
and unclipped. Original two PDFs remain unchanged.
