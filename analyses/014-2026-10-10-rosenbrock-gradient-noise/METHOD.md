# H013: additive estimation noise in Rosenbrock gradients

Pre-execution contract, 2026-10-10. The user explicitly resolved the original
formula: **keep Rosenbrock and use g_t = grad f(theta_(t-1)) + xi_t**, not the
literal theta gradient of a quadratic. They selected **one seed for a first look**.
Noise xi_t is independent N(0,sigma^2 I_2) across optimizer updates/coordinates.
Use sigma=0,0.01,0.1,1,10 and noise seed1234, NumPy Generator(PCG64), generating
one standard-normal array of shape(2000,2) per run and multiplying by sigma.
Reuse the exact same standard-normal sequence across all nonzero sigmas/LRs.
Different sigma values change paths, so only the noise draws, not subsequent
analytic gradients, are paired. Sigma is an absolute standard deviation in
gradient-coordinate units, not a relative percentage of gradient norm.

Reuse sigma0 controls Runs038-040 read-only, with no duplicated deterministic
replicates. Execute Runs041-052 (four nonzero sigmas times three LRs), 2,000
updates each. Match start(0,1), f=(1-x)^2+100(y-x^2)^2, optimum(1,1), betas
(0.9,0.95), epsilon1e-8, weight decay0.1 on both coordinates, constant LR
1e-2/sqrt(1e-3*1e-2)/1e-3, zero moments, full2D CPU float64, no clipping,
no early stopping, no dataset or cloud execution. Initial loss101 is unchanged.
Record config/source/dependency hashes before launch; preserve every prior input.

Raw g now means the **noisy estimator used by AdamW**, before optimizer
transformations. Retain the exact analytic signal h_t and sampled xi_t separately,
plus standard-normal draws, all2,001 positions/losses/true post-gradient norms,
2,000 stored first/second moments, adaptive directions, actual displacements and
complete geometry for raw/momentum/adaptive/displacement/signal/noise streams.
True objective has no sampled observation noise or L2 penalty. The descriptive
joint stationarity test still uses true f<=1e-8 and true post-gradient norm<=1e-4,
with the final100 positions required together; do not confuse noisy norm with it.

For any vector stream z: A=sum||z||, B=||sum z||, B/A; adjacent cosine uses
consecutive recorded vectors. First cosine and zero-vector pairs are undefined;
A=0 gives undefined ratio, positive A with B=0 gives zero. No added epsilon,
clipping or weighting. Preserve energy E=sum||z||^2, exact all-pair interaction
I=2 sum_(r<s)<z_r,z_s>, lag-one interaction L=2 sum<z_(t-1),z_t>, and I-L.
B^2=E+I holds pathwise. A^2-B^2 is the triangle gap, not itself I.
The exact momentum-prefix identity uses the noisy inputs:
sum m = sum g - beta1/(1-beta1) m_last, since m_initial=0.

Noise decomposition at every prefix, with H=sum h and X=sum xi:
B_g^2=||H||^2+||X||^2+2<H,X>;
E_g=E_h+E_xi+2 sum<h_t,xi_t>;
I_g=I_h+I_xi+2(<H,X>-sum<h_t,xi_t>).
Retain cross terms. Although current noise is conditionally centered, later
signals depend on earlier noise along an adapted trajectory; do not assume
aggregate signal/noise cross terms vanish. For the independent noise stream,
E[||X||^2]=E[E_xi]=2*t*sigma^2 is an ensemble expectation, not a pathwise equality.

One separate nine-panel PDF per sigma in this analysis: full objective
contours/three trajectories, loss, raw/momentum adjacent cosine (trailing100
finite values, first at101), raw/momentum norms, raw A/B, momentum A/B, raw
B/A and momentum B/A. Keep linear norm/A/B Y, log loss Y and full trajectories;
use one common landscape domain across the sweep. Label raw as noisy and sigma
explicitly. There is one path per sigma/LR, hence no replicate bands or uncertainty
estimate. Do not interpret a first local y reversal under noise as the same
coherent valley-entry event; the previous turn annotations are not used here.
Add a sigma-comparison figure and a squared-sum/energy/interaction decomposition
figure per sigma to expose the requested identities and increasing noise.

Presentation addition requested after execution: a 5x3 norm grid, one LR per
column and one sigma per row, with raw and stored-momentum norms at every step.
Use common log Y limits to show early/late scales and compare matched cells;
no smoothing or thinning. This addition changes presentation only.

Verify before acceptance: mechanics of additive noise and zero/cancellation
cases; sigma0 engine reproduces every saved baseline array exactly; same seed
and standard draws across conditions; xi=sigma*z and raw=signal+xi; analytic
signal/objective against SciPy; full exact replay; independent PyTorch AdamW fed
every captured noisy gradient; all-stream norms/prefixes/cosines/temporal identities,
momentum prefix identity, decay displacement and telescoping; all artifact
sizes/hashes and dependency identities. Check every rolling mean and figure
metadata, text/page integrity and actual rendered layouts before committing.

Local execution is authorized. Plan24,000 new updates, roughly seconds of
simulation/saving plus verification/plotting, and about20 MB of additional
histories (qualified extrapolation from prior complete2D runs; refresh measured
values). No billable resource. Persist manifests/progress every200 updates and
final optimizer/RNG information. Retain raw histories locally, ignored in Git;
compact configs/receipts/reductions/observations/publication PDFs are versioned.

Interpretation: an exploratory single realization paired across sigmas/LRs.
No claim of monotonic loss or geometric response to sigma, convergence,
stability, a causal alignment benefit, or LLM transfer is pre-specified.
