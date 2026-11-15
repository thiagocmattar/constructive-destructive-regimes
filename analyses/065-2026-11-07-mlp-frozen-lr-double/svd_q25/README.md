# Q25 and the dimension of representation movement

Ongoing Analysis065 / H053 / [Run297](../../../runs/297-2026-11-15-cap-q25-temporal-svd/README.md).
The user requested a direct SVD comparison, MLP first and then ResNet and
Transformer, using owned Spark and keeping the manuscript/consolidation intact.
Interpretation and figure captions are in
[Observation017](../observations/017-q25-representation-svd.md).

## Matched comparison

Unchanged Adam versus the distributed Q25 cap, starting from the original
authenticated Run293/294/295 forks at states 512/256/544. The first 64 incoming
changes are preselected, with first/last 32-change windows as sensitivity checks.
This is an early post-treatment comparison, not a selection of the historical
largest ResNet excursion. Original full training objectives, saved moments and
clock, frozen cap calibration, and fixed diagnostic probes are retained.

The source runs saved scalar motion histories and sparse complete states, not
dense activation histories at every update. Run297 therefore reconstructs these
short continuations on Spark. MLP can be compared exactly with its Spark history;
ResNet and Transformer originated on other GPU architectures, so historical
agreement is measured rather than assumed. Within each new pair both arms use
the same hardware and deterministic runtime.

## Measurements

For each site, D has one row per incoming representation change and one column
per fixed sample/position/feature coordinate. No time centering, feature
averaging, or per-row normalization is applied. Exact FP64 temporal Grams give
the squared singular values. Report rank95/99, participation and entropy ranks,
spectral shares, absolute squared movement, coordinate RMS, reversal and net
displacement retention. All comparisons use equal time windows and coordinate
sets within an architecture.

Cross-arm Grams measure how much Q25 movement lies inside the baseline's fixed
right singular subspace covering 95% of baseline movement. This compares spatial
patterns across interventions, unlike the paper's cross-layer temporal overlap.
The baseline has residual movement outside that subspace by construction. Q25
movement outside it does not imply a direction absent from every baseline mode,
increased model capacity, or greater learning. Report both relative fractions
and absolute residual energies.

For baseline B and capped C change matrices, retain `K_B = B B^T`,
`K_C = C C^T`, and `G = B C^T`. If `(lambda_i, u_i)` are baseline temporal
eigenpairs and k is baseline rank95, capped energy inside its right subspace is
`sum_{i<=k} ||u_i^T G||^2 / lambda_i`. Outside energy is `trace(K_C)` minus
that quantity. Baseline outside energy is the sum of its eigenvalues after k.
No explicit P-dimensional right basis or random projection is needed.
The supplementary right-subspace overlap is `||V_B,k^T V_C,k||_F^2 / k`, with
common `k = min(rank95_B, rank95_C)`. It compares equal-sized leading subspaces
(one for identical subspaces), rather than pairing individual singular vectors.

With normalized eigenvalues p, participation rank is `1 / sum(p^2)` and entropy
rank is `exp(-sum(p log p))`. Rank95 is the smallest prefix covering 95% of
total squared movement. Retention is `||sum_t delta_t|| / sum_t ||delta_t||`.
Adjacent reversal excludes pairs with a zero change norm; effective dimensions
are undefined for zero total motion. All measured site/window energies here
are nonzero. Exact refers to retaining every coordinate in the Gram reduction,
subject to the declared floating-point precision.

## Outputs and reproduction

- `metrics.csv`: all sites, both arms, all three windows; no site selection.
- `comparisons.csv`: matched energy ratios, rank changes and fixed-baseline
  subspace projections for every site/window.
- `<case>-q25-svd.pdf`: overview, all-site cumulative spectra, window sensitivity.
- `comparison-q25-svd.pdf`: one-page comparison of all-site rank95 and output
  spectra across the three matched model cases.
- `verification.json`: independent compact spectral/diagonal checks and direct
  dense-SVD/right-projection checks at manageable readouts.
- `provenance.json`: coordinate layouts, capture times, historical trajectory
  qualification, full-training loss context and source hashes, retained alongside
  the compact results.
- `<case>-retrieval.json`: checksum-verified compact retrieval and raw retention.
- `dependency-audit.json` and its retrieval receipt: 81 source-file comparisons
  against the authenticated original Runs293-295 manifests.
- `raw/<case>/`: ignored local compact source evidence; complete FP32 activation
  histories and recovery states remain on owned Spark with SHA-256 inventory.

Run-local `01_capture.py` and `02_reduce.py` produce the evidence.
`70_collect_q25_svd.py <case>` retrieves it;
`71_report_q25_svd.py` verifies and rebuilds the tables and figures.
Run-local `06_verify_dense.py` supplies the independent direct-SVD check;
`07_verify_and_collect.py` waits for successful queued completion, verifies,
retrieves and regenerates the local report.
Each PDF's page 1 compares rank95, total motion, output spectra, residual motion
outside the baseline95 subspace, and full training loss. Page 2 retains every
site's normalized spectrum. Page 3 compares matched 32/32/64-update windows.
Sources and figure hashes are in `figure-manifest.json`.

Coverage is one initialization per architecture, fixed reference sets and
short post-fork windows. Time steps and units are not independent replicates.
No generalization or causal mechanism of loss spikes is inferred.
