# Observation002: Analysis009 collector geometry on Rosenbrock

Sources/coverage: all 10,000 g,m,d,u vectors and prefixes in the four owning
history.npz files; hashes/receipts in verification.json and figure_manifest.json.
Generator02_plot.py produces figures/squared_geometry.pdf and norm_ratios.pdf.
Every vector is measured over both coordinates. Neither figure smooths or
subsamples updates. Log displays mask nonpositive/undefined values without
epsilon substitution; raw arrays retain them. The pilot has no zero vector
denominator; first cosine remains undefined. All output pages passed visual QA.

Caption, squared geometry: **Rosenbrock: Analysis009 squared cumulative
geometry.** Raw gradient and stored momentum form the two columns; rows show
A², B² and A²−B² for four conditions. A=sum of norms, B=norm of cumulative sum.
Each panel uses its own log Y range. Zero triangle gap at the first update is
omitted from logarithmic display. The nonnegative triangle gap is distinct
from signed pair interaction I=B²−sum of squared norms.

Caption, ratios: **Paired collector norm ratios and stationarity.** Every
update's ||m||/||g||, ||d||/||g||, ||d||/||m|| and post-update objective gradient
norm is plotted on log Y axes; one line per condition. The final panel includes
the gradient threshold 1e-4, which additionally requires f<=1e-8 for a joint hit.

Result: in Run028, raw A=6440.185843 and B=1581.354803, B/A=0.245544902;
stored momentum A=2466.946494 and B=1581.376057, B/A=0.641025681.
The near-equality of B and smaller momentum A is inspectable without clipping
or noisy batches. Indeed every prefix obeys the exact no-clipping recurrence
sum m = sum g − beta1/(1-beta1)*m_t with zero initial moment. This identity
is explicitly verified, not inferred from plotted endpoints. At beta1=0,
stored momentum equals raw gradient exactly.

With zero decay and constant LR, u=-LR*d: Run028 adaptive and displacement
B/A agree (0.219650795), and their adjacent cosines agree to numerical precision.
With decay present the actual displacement includes an additional originward
term, so adaptive directions must remain separately named and measured.
Cumulative displacement B is exactly ||theta_t-theta_0|| up to verified
floating-point residual, even when path length A keeps growing near the optimum.

Uncertainty/limits: exact reductions of these deterministic histories, not an
estimate over seeds, starts or functions. Low B/A can reflect directional
cancellation or orthogonality and does not by itself imply a harmful loss effect.
Momentum's A/B changes do not establish information preservation or a causal
benefit of alignment. This pilot has no per-sample/intra-batch decomposition.
