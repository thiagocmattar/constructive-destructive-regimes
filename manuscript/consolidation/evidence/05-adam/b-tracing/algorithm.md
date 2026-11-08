# Algorithm 1 — fixed-probe architectural oscillation tracing

- **Status:** consolidated notation, ready for integration; original algorithm remains in the local report.
- **Inputs:** qualified trajectory, fixed ordered probe, architecture stages, outgoing window [a,b].
- **Outputs:** per-stage motion/cosine/RMS, temporal spectra and signed coefficients, coverage and verification records.

1. Fix matched samples, coordinate order, stochastic context and normalization groups; ensure measurement does not alter training.
2. Capture representations at states a−1 through b+1, losses, and raw gradients a−1 through b. The extra predecessor supplies the first cosine.
3. Compute each outgoing flattened activation change, its adjacent cosine and coordinate RMS. Retain every sample/token/feature coordinate.
4. Measure raw-gradient cosine separately. Do not difference gradients or identify them with Adam increments.
5. For each stage, form the FP64 temporal Gram on rows a…b, with no centering or primary row normalization.
6. Recover singular energies and signed temporal coefficients. Fix mode signs once; compare temporal subspaces only within numerical ranks.
7. Inspect the output and move backward through stages and residual branches, comparing direction, magnitude and timing.
8. Verify observer invariance, source identity, zero norms and reductions. Report probe/window coverage, cost and causal limits.

- **Cost:** rolling cosine/RMS O(P) per stage/update; full-window Gram O(T²P), eigendecomposition O(T³). Retrospective SVD is not an online predictor.
- **Equations:** [definitions and matrix sizes](equations.md). [Editable TeX](algorithm.tex).
- **Source:** [original formal method](observations/63-METHOD.md), [report](report/instability-tracing-report.pdf).
