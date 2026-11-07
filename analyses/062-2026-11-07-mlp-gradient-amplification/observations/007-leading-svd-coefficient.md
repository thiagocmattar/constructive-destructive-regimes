# Leading functional coefficient: direct projected reversals

The requested diagnostic confirms repeated back-and-forth motion along the
leading fixed functional axis. This is a post hoc analysis of retained Run239
outputs; no training or remote execution was needed.

- [Figure06](../figures/06-leading-svd-coefficient.pdf)
- [Generator](../08_leading_svd_coefficient.py)
- [All plotted values and projected positions](../leading_svd_coefficient_values.csv)
- [Numerical checks and source hashes](../leading_svd_coefficient_verification.json)
- [Existing SVD definition and provenance](006-svd-output-change-heatmap.md)

## Definition and temporal indexing

Use exactly the existing uncentered SVD of the **output differences**, not the
outputs themselves. For $t=448,\ldots,800$, flatten all 5,000 examples and all
10 outputs into $F_t\in\mathbb R^{50000}$. The rows of the fitted matrix are
$D_{t,:}=\Delta F_t=F_{t+1}-F_t$. Its leading unit right singular vector $v_1$
is fixed across the entire window, with its largest-absolute coordinate positive.
Define

$$
\alpha_t=\langle F_{t+1}-F_t,v_1\rangle,
\qquad p_t=\langle F_t-F_{448},v_1\rangle,
\qquad \alpha_t=p_{t+1}-p_t.
$$

Thus $\alpha_{t-1}\alpha_t<0$ is a reversal of movement along this same axis
on two consecutive updates. A global sign change of $v_1$ does not change this
criterion. No refitting or sign selection occurs between steps. The plotted
coefficient is raw; it equals heatmap row 1 multiplied by $\sqrt{5000}$.
There is no example averaging or coordinate standardization.

The loss at $t$ precedes update $t\to t+1$. Consequently, a peak at state474
is entered by $\alpha_{473}$, and $\alpha_{474}$ measures the outgoing motion.
The source includes state801 to measure $\alpha_{800}$. Counts use only the
352 adjacent coefficient pairs within the displayed window; no coefficient at447
is included. Exact zeros would be reported separately, with no reversal counted;
none occur. No magnitude threshold or smoothing is applied.

## Results

| Coefficient interval | Sign reversals / adjacent pairs | Fraction |
|---|---:|---:|
| 448--800 | 306 / 352 | 86.93% |
| 465--474 | 9 / 9 | 100% |
| 512--522 | 10 / 10 | 100% |
| 465--485 | 17 / 20 | 85.00% |
| 512--529 | 15 / 17 | 88.24% |

For example, $\alpha_{470},\ldots,\alpha_{475}$ are approximately
$8.87,-12.27,15.79,-18.35,18.32,-14.03$. The alternating movement grows in
magnitude approaching the state474 loss peak, then its envelope declines.
The two zoom windows are previously studied spike episodes, not independent
replications or newly selected maximizers of this statistic.

To check whether tiny coefficients dominate the reversal count, also weight
each adjacent pair by $w_t=\alpha_{t-1}^2+\alpha_t^2$. The descriptive ratio

$$
\frac{\sum_{t=449}^{800}w_t\,\mathbf{1}[\alpha_{t-1}\alpha_t<0]}
{\sum_{t=449}^{800}w_t}=0.976211
$$

is 97.62%. This is an overlapping-pair coefficient-energy weighting, **not**
the fraction of total network displacement energy that reverses. Interior
coefficients enter two pair weights. It introduces no amplitude cutoff.

The leading axis accounts for **25.1689%** of the full window's squared
output-change energy. It is the strongest single fixed axis, but does not
describe the entire trajectory. These results directly establish projected
oscillation; they do not by themselves establish why the loss spikes, that
every component reverses, or that movement returns to the identical network
output vector. The previously measured full-vector cosine supplies complementary
information. This full-example temporal SVD differs from the earlier per-event
5000-by10 class-space SVD and its approximately93% leading share.

## Sources and verification

Use unchanged Run239 `outputs.pt` and `boundaries.jsonl`, plus the existing
`artifacts/svd-output-changes/patterns.pt` basis in Analysis062. All source hashes
are checked against the existing heatmap audit. NumPy projects the raw FP64
differences independently of the stored SVD coefficients; maximum absolute
error is3.74e-14 or less. Projected-position differences agree to the same
tolerance. The generator verifies unit norm, first-mode energy share and all
353 coefficients against the original heatmap CSV. Sources, generator, CSV and
PDF have SHA256 hashes in the verification record. The PDF was rendered and
visually inspected. Dense source arrays remain in ignored artifacts.

One seed, one selected window, fixed training examples, and a basis fitted using
the entire window. These are descriptive counts, not independent Bernoulli
trials; no IID confidence interval or causal inference is attached. Other
windows or seeds can have different leading axes and reversal rates.

## Caption

**Signed leading SVD coefficient exposes projected oscillation.** (a) Full
training loss over states448--800. (b) Signed projection of each outgoing raw
output difference onto the leading fixed full-example SVD axis. Red and blue
indicate opposite projected motion;306/352 consecutive coefficient pairs reverse
sign. (c,d) Previously studied spike episodes around loss peaks474 and522,
shown at individual-step resolution. Gray bands in(a,b) locate the zoom windows;
dashed vertical lines in(c,d) mark peak-loss states. Coefficients are unscaled,
and the axis explains25.17% of full-window squared change energy.
