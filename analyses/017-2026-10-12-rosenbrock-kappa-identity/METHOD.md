# H016: gradient-identity curvature versus the Rosenbrock Hessian

The user requests Analysis016's analysis with

    kappa_proxy[t] = (1 - rho[t] * ||g[t]|| / ||g[t-1]|| + epsilon[t]) / eta.

Only the diagnostic changes. Reuse immutable Runs055-057: deterministic
Rosenbrock, start(0,1), 2,000 AdamW updates, LR0.01/sqrt(0.001*0.01)/0.001,
betas(0.9,0.95), optimizer epsilon1e-8, decoupled decay0.1, zero initial
moments, no clipping, CPU float64. No new trajectories, data, seeds, cloud or
bills. Histories contain every vector, position and exact objective Hessian.
This is a new independent analysis; the next run ID remains058.

The supplied restatement defines epsilon as normalized stochastic gradient
noise. Here gradients are exact and deterministic, so epsilon[t]=0. The
optimizer's numerical epsilon is a separate quantity and is not added to this
formula. Decoupled decay is not included in the objective Hessian.

Indexing: update t uses g[t]=grad f(theta[t-1]). At t=2..2000, compute rho from
g[t-1],g[t], and use the previous update's constant base learning rate eta.
Compare with H(theta[t-2]), the Taylor-expansion point in the restatement.
Retain lambda_min/max at that same point and the exact directional curvature

    kappa_direction[t] = g[t-1]^T H(theta[t-2]) g[t-1] / ||g[t-1]||^2.

All pair series have 2,000 entries indexed t=1..2000; the first is undefined.
Either zero adjacent norm makes the literal cosine identity undefined; do not
insert a denominator epsilon, clip to the Hessian spectrum, or add a floor.
No zero adjacent norms occur in these sources. All 1,999 defined pairs per LR
remain, including the terminal pair. Original loss/Hessian/trajectory cover all
2,001 positions. Keep original first-y-turn references81/228/655 at their
position clock; those geometric references are not estimator thresholds.

The manuscript's derivation assumes plain SGD and a first-order Taylor step.
AdamW follows a different displacement. Evaluate the requested expression as
a proxy and show the exact Rayleigh quotient as well as lambda_max. Even under
plain SGD, directional curvature need not equal lambda_max. Quantify the exact
algebraic discrepancy along these AdamW steps: with u=g[t-1],
delta=theta[t-1]-theta[t-2], and r=g[t]-u-H delta,

    kappa_proxy = kappa_direction
                  - u^T H(delta+eta*u)/(eta*||u||^2)
                  - u^T r/(eta*||u||^2).

The first correction is update mismatch relative to plain SGD; the second is
the nonlinear Taylor remainder. These are descriptive exact decompositions for
epsilon=0, not interventions or attribution of a causal benefit.

Primary comparison PDF: three LR columns; top row unsmoothed proxy, lambda_max
and exact directional curvature; bottom row complete100-pair trailing means
of each. Use common linear Y limits within each row, including every finite
value; no zoom, clipping, normalization, thinning or uncertainty band. The
first full mean is at t=101. Compute the identity from unsmoothed quantities
before averaging. A second PDF retains Analysis016's full trajectory/loss/
cosine/norm/A/B/B/A/Hessian layout and adds proxy and directional-curvature
panels. Only cosine/proxy panels use100-step means; faint raw proxy remains.

Report Pearson association over all aligned pairs and over100-pair means,
late100-pair arithmetic means (t=1901..2000), signed/absolute errors and
outside-spectrum counts. Correlations describe deterministic time series;
they provide no independent samples, p-values, uncertainty or validation of a
curvature estimator. Their sensitivity to smoothing is explicit.

Verify hashes and original source receipts, analytic gradients/Hessians at
every source point, indexing, direct dot-product versus cosine identity,
Rayleigh spectral bounds, exact discrepancy decomposition, a quadratic-SGD
case and zero-norm/noise conventions. Independently recompute complete-window
means and summaries, check every rendered line against saved arrays, render
both PDFs with Poppler and inspect them. Record scripts, input/output sizes,
SHA-256, runtime, retained local storage and figure captions in this analysis.

The follow-up explicitly requests an additional single-panel PDF while keeping
the original comparison unchanged. Plot eta*kappa_proxy, then a complete100-pair
trailing mean, with the three LRs overlaid. This dimensionless scaling cancels
the formula's explicit1/eta factor. Save04_plot_eta_overlay.py, exact plot data
and a manifest proving the original comparison SHA-256 unchanged. No optimizer
or source-gradient change is introduced; three final PDFs are retained.

Formula provenance: formula_source_snapshot.md is an exact byte snapshot of
the supplied restatement at initial calculation, SHA-256e689fd9ec071a431026b422aa345999730a3860bffa87ff05799721ebe2d8ddf.
The user edited the manuscript concurrently; its original recorded hash was
used to verify the recovered local snapshot. Source manuscript edits are
preserved. Reduction/verification now read the frozen snapshot; numerical
inputs, derived arrays and all previously generated PDFs remain unchanged.
The metadata correction is recorded in provenance_snapshot.json.
