# Design and operational definitions

H017 asks how Analysis017's 100-pair trailing means of the learning-rate-scaled
gradient identity change with additive Gaussian gradient-estimation noise.
This is exploratory postprocessing, without a directional prediction or new
optimizer execution. Sources are Analysis014's immutable Runs038-052: five
noise levels times three learning rates, including the existing zero-noise controls.

Vary sigma = 0, 0.01, 0.1, 1, 10 and constant learning rate eta = 0.01,
0.0031622776601683794, 0.001. Match Rosenbrock
f(x,y) = (1-x)^2 + 100(y-x^2)^2, initial point (0,1), 2,000 updates,
AdamW betas (0.9,0.95), numerical epsilon 1e-8, decoupled weight decay 0.1,
zero initial moments, no clipping, CPU float64, and fixed update budget.
For nonzero noise, seed1234 PCG64 standard-normal draws are identical at
all noise levels and learning rates. The coordinate standard deviation sigma
has absolute gradient units; it is not a fixed relative noise fraction.

Let s_t = gradient f(theta[t-1]), xi_t be the stored additive noise and
g_t = s_t + xi_t the raw gradient actually supplied to AdamW. For t=2..2000:

\[
q_t = 1-\rho_t\frac{\|g_t\|}{\|g_{t-1}\|},\qquad
\varepsilon_t = \frac{g_{t-1}^{T}(\xi_t-\xi_{t-1})}{\|g_{t-1}\|^2},\qquad
\eta\widehat\kappa_t=q_t+\varepsilon_t.
\]

The left column repeats the previous observable q_t with epsilon omitted.
The right column implements the identity with its realized, known noise term.
Both columns equal the old normalized overlay at sigma0, verified against
Analysis017's recorded arrays. Epsilon here is the restatement's noise term,
not AdamW's numerical epsilon. Neither zero expectation of fresh noise nor
unbiased gradient estimation justifies setting this realized term to zero.

Compute each pair first, then its complete finite trailing arithmetic mean over
100 pairs. This is an average of per-pair ratios, not a ratio of averaged norms
and cosines. Index1 is undefined; if either adjacent raw gradient norm is zero,
the pair is undefined. No denominator floor, clipping, trimming, winsorization,
thinning, partial windows or seed averaging. Any undefined pair invalidates
that full window. Actual coverage: 1,999 pairs and 1,900 means per trajectory,
first complete mean at t=101. All finite points appear in the figure.

Left Y limits are common across all five sigmas. Right Y limits vary by row
to make the much smaller corrected values visible; all axes are linear.
Three LR colors/styles match Analysis017. Noise can induce immediate small
trajectory reversals, so no first-y-turn lines are transferred from deterministic
runs or redefined as equivalent stochastic navigation events.

## Interpretation and exact cross-check

The derivation assumes plain SGD and a first-order gradient approximation.
These histories use AdamW, including momentum, adaptive scaling and decay.
Adding epsilon cancels the explicit noise difference algebraically:

\[
q_t+\varepsilon_t =
-\frac{g_{t-1}^{T}(s_t-s_{t-1})}{\|g_{t-1}\|^2}.
\]

It does not remove the noisy path or turn AdamW into SGD. With
u = g[t-1], H = Hessian f(theta[t-2]),
delta = theta[t-1]-theta[t-2], and
r = s[t]-s[t-1]-H delta, the exact diagnostic decomposition is

\[
q_t+\varepsilon_t=
\eta\frac{u^T H u}{\|u\|^2}
-\frac{u^T H(\delta+\eta u)}{\|u\|^2}
-\frac{u^T r}{\|u\|^2}.
\]

All three terms are retained and independently checked. A synthetic noisy
quadratic with plain SGD recovers eta times directional curvature exactly,
testing the epsilon sign and indexing. Actual AdamW mismatch persists.
Neither plotted column estimates the largest Hessian eigenvalue directly.

One paired seed has no ensemble confidence intervals or estimated sampling
uncertainty. Overlapping windows are dependent. Tiny preceding raw norms can
amplify ratios; all are retained. Noise changes both the estimator and subsequent
trajectory, and weight decay effects scale with LR. Do not infer a stability
threshold, monotone ensemble law, asymptotic convergence or LLM transfer.

## Sources and regeneration

- Frozen `cohort.json` copies Analysis014's source membership.
- `results.json` records each source history/config/verification SHA-256,
  derived-history sizes/hashes, metric/config/code identities and summaries.
- The formula's frozen source is Analysis017's `formula_source_snapshot.md`;
  its SHA-256 and the old overlay data SHA-256 are recorded here.
- Original run histories and original Analysis017 PDFs are preserved.
- `00_compute.py`: derived histories and results, local CPU only.
- `01_verify.py`: independent pair algebra, SciPy objective derivatives and
  Hessians, exact decomposition, all full windows, paired draws, sigma0
  equivalence, synthetic noisy SGD and undefined behavior.
- `02_plot.py`: PDF and exact plotted means in `plot_data.json`.
- `03_record.py`: plotted-array provenance, complete coverage, bounds and PDF
  layout checks; render to ignored scratch. Inspect the render before using
  `--record-visual-review` to mark the review complete.

Execute the four numbered Python scripts in order. The final script's review
flag records a human/agent visual inspection; it does not perform one itself.
Derived NPZs (2,655,983 bytes) are retained locally outside Git. No cloud
resource, billable job, dataset download or optimizer run is created.

## Additional 250-pair rendering

`config_mean250.json` specifies a separate visualization using complete finite
trailing means of250 adjacent-pair values. `04_plot_mean250.py` reads the same
saved unsmoothed pair metrics and generates a new suffixed PDF, data JSON and
manifest. It does not change the original computation, input config, source
histories,100-pair PDF,100-pair plotted data or original manifest. The first
complete mean is at t=251; each curve has1,750 complete means at t251..2000.

Layout, metrics, LR encodings, noise pairing and axis-scale policy match the
original: common linear Y limits on the left, separate linear ranges by row
on the right. Bounds are recomputed from the250-pair values to show every
finite mean. `05_verify_mean250.py` checks all52,500 plotted means against
direct pair metrics reconstructed from the immutable raw/signal vectors,
source hashes, serialization, PDF text/bounds and preservation of the original.
It also renders the new PDF; inspect it before recording visual review with
`--record-visual-review`. Longer smoothing reduces short-scale fluctuations
and spreads transition timing; it adds no samples, seeds or independent evidence.
