# Local displacement caps: technical report and manuscript figures

## Outcome and scope

The eight-page [LaTeX report](../report/local-displacement-cap-report.pdf) consolidates
local affine-displacement caps as a stabilization intervention motivated by
representation/destructive-regime tracing. It contains motivation, matrix
dimensions, the per-update algorithm, calibration, entry/distributed scopes,
full training objectives and probes, results, replication, cost and limitations.
The [editable source](../report/local-displacement-cap-report.tex) is self-contained
with three generated table inputs and three companion vector figures.

The terminology clarification defines an affine operation as weighted sum plus
bias, and "direct affine response" as the local preactivation change caused by
the proposed weight/bias update with inputs held fixed. A one-neuron numerical
example and the algorithm use this explanation; incoming representation changes
and subsequent nonlinear transformations remain explicitly separate.

The notation revision follows the active manuscript's
[Adam section](../../../manuscript/draft/sections/06-adam-diagnostics.tex) and
[raw-gradient diagnostics](../../../manuscript/draft/sections/03-methods.tex).
It reserves $m_t$, $v_t$, and $d_t$ for Adam's first moment, second moment, and
adaptive direction, and $\rho_t$ for consecutive raw-gradient alignment.
The previous weight/bias symbols $U_t,v_t$ become
$\Delta W_t^{\mathrm{Adam}},\Delta b_t^{\mathrm{Adam}}$; the previous $D_t$
becomes $\Delta Z_t^{\mathrm{local}}$, the proposed preactivation change at fixed
inputs. Superscript `cap` denotes accepted increments. The whole-parameter
increment ratio is $\gamma_t$. The Adam equations, cap, numbered algorithm,
actual-change identity and all three companion figures use the same conventions.
The extra report page preserves readable typography while defining the optimizer
quantities and distinguishing update indices from plotted states.

**Feedback 001 / Revision 001:** the [numbered archive and response](../report/feedback/001/README.md)
preserve the supplied annotated PDF, every annotation, source hashes and all
dispositions. The revision explains fixed input correspondence and flattening,
uses event-window tracing evidence, defines every local matrix and the per-unit
score, distinguishes one scalar module threshold from per-unit factors, and
clarifies update-step calibration and which weight row/bias is scaled. Practical
cost is one additional probe forward plus local affine products and checks,
not two full-network forwards to obtain the local score. Online changing-data
use remains future work. Ten complete comments are addressed; the unfinished
"Or maybe we coul" note is retained without an inferred request. Only explanatory
text and layout changed; all numerical results and standalone figures are preserved.

Only cap variants and their unchanged-Adam controls appear. The earlier broad
exploration is preserved in observation 008 and Figures 06-08. This report adds
no experiments, no cloud spending, and no run/analysis/hypothesis IDs.

The common distributed q50 rule lowers mean training loss by 22.85%, 16.22%,
and 26.00% for MLP, ResNet and Transformer in their respective windows. MLP
initializations 1/2 improve 19.43%/32.08%. Negative representation-change
alignment persists or becomes more frequent, and terminal test loss is worse
under this cap in every setup. Stabilization here concerns measured loss
excursions together with training progress, not a convergence guarantee.

## Source records and reduction

Sources are the verified `transfer_comparison.json` and follow-up comparisons
for Runs 272, 273, 274, 282 and 283, plus `transfer_summary.json`. Entry figures
use Runs 269/270/271 (q75); distributed figures use Runs 272/273/274 (q50);
replication uses Runs 272/282/283. Tables additionally retain MLP entry q50,
MLP distributed q75, and ResNet distributed q75. No other intervention result
is included in the report. Source and generator hashes are in
[local_cap_report_assets.json](../local_cap_report_assets.json).

`34_local_cap_report_assets.py` checks existing reduction hashes, exact plotted
arrays, native state sequences and all 12 selected table rows against independent
reductions of the loss histories. The tables report full-window mean, last-32
mean, maximum and total positive loss variation relative to each worker's own
untreated control. The seed table also reports complete 10,000-image terminal
test loss. The report's ResNet/WikiText-2 test comparisons retain full cached
splits (10,000 images / 245,550 supervised test tokens).

Manuscript update $t$ starts at $\theta_{t-1}$ and ends at $\theta_t$, with
$g_t=\nabla\mathcal L(\theta_{t-1})$ and
$\delta_t^\ell=\operatorname{vec}(A_t^\ell-A_{t-1}^\ell)$.
The horizontal axis retains completed updates $s$, with no shifts to recorded
arrays. At $s$, loss is $\mathcal L(\theta_s)$, the raw gradient is $g_{s+1}$,
and outgoing representation movement is $\delta_{s+1}^\ell$.
The plotted raw cosine $\rho_{s+1}$ compares gradients at states $s$ and $s-1$;
the representation cosine $\rho_{s+1}^\ell$ compares changes $s\to s+1$ and
$s-1\to s$. An extra state after the loss window supplies its last outgoing
diagnostic. The first intervention is update $t_0$; the eight calibration
updates are $t_0-8,\ldots,t_0-1$. These are notation conversions, not altered
calibration or evidence. All sample/token/spatial coordinates are retained.
Zero-denominator cosines remain undefined gaps, with no smoothing or interpolation.

## Figure captions

**Figure 09: entry local displacement caps, q75.** Three model/dataset columns
compare each cap with its own unchanged Adam control. Windows are MLP 448-800,
ResNet 264-576, and Transformer 552-928. Ten rows show loss, eta-kappa-hat,
raw-gradient cosine/norm, early representation cosine/RMS, output cosine/RMS,
actual/native whole-update norm ratio and the mean fraction of targeted units
modified. Early means first MLP tanh, ResNet conv 1 output, or first Transformer
attention output. The modified fraction averages over targeted modules; it is
not the fraction of all parameters modified.

**Figure 10: distributed local displacement caps, q50.** Same ten diagnostics
and column definitions, using the wider target scopes defined in observation
010. MLP extends through 1152; the other windows remain unchanged. Transformer
caps target 2.91% of all parameters, so a whole-update norm ratio near one can
coexist with frequent modification of the targeted units. Its baseline is the
same-worker RTX 5090 control, not the RTX 4090 control in Figure 09.

**Figure 11: MLP initialization replication.** Three columns show initializations
0/1/2 over states 448-1152, comparing only distributed q50 with each seed's
unchanged Adam. Rows show training loss, consecutive output-change cosine and
output displacement coordinate RMS. Seed 0 selected the candidate; seeds 1/2
use the same rule without retuning.

All figures follow the manuscript's 7.16-inch width, Times New Roman/STIX
typography, vector lines, restrained grid and shared legend convention.
**Loss, gradient norms and RMS use logarithmic axes. Eta-kappa-hat uses a linear
axis, as explicitly requested.** Each model/seed column has independent
magnitude scales; cosine limits are -1 to 1. No signed-log axis is used.

## Interpretation and uncertainty

This is an amplitude intervention on local affine responses, informed by the
earlier tracing method. It neither computes a temporal SVD nor directly
suppresses a singular direction or negative cosine. The report derives the
incoming-input terms excluded from the local cap and distinguishes its direct
per-unit RMS bound from a bound on all examples or total network-output change.

Entry q75 versus distributed q50 changes both scope and quantile. The Transformer
comparison also changes GPU architecture, whose native controls differ. Effects
are therefore paired within workers; no isolated Transformer depth effect is
claimed. The report includes the matched-quantile MLP and ResNet comparisons.
These are selected deterministic full-objective continuations, three MLP
initializations and one initialization per larger model, without a powered
uncertainty estimate, stochastic-minibatch validation or an optimal-quantile claim.
Held-out losses are terminal measurements, not trajectories; their deterioration
is retained prominently in the report.

## Reproduction and verification

From the repository root:

```powershell
python analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py
python analyses/065-2026-11-07-mlp-frozen-lr-double/35_build_local_cap_report.py
```

The asset generator checks 61,450 plotted values and preserves earlier Figures
01-08 byte for byte. The two-pass pdflatex build passes with a numbered algorithm,
eight pages, no unresolved references, no overfull boxes and no out-of-page text.
All eight report pages were rendered at 120 dpi and visually reviewed; the three
unchanged standalone figures retain their preceding verified review. See [build verification](../local_cap_report_verification.json)
and [visual review](../local_cap_report_visual_review.json). Scratch PNGs and
LaTeX compilation files remain ignored.
