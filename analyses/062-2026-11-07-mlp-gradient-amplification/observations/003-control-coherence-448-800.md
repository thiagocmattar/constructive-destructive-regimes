# Seven-row control figure, states448 through800

Run238 extends H042 / Analysis062 to produce the user's requested single-column
figure. It replays the unchanged Adam control, with no optimizer intervention,
and measures the newly requested norm-based intra-batch coherence. Earlier
figures and the squared-energy coherence definition remain unchanged.
Run239 adds the requested final output-change alignment row using an exact
replay and one additional unchanged Adam update for its endpoint.

## Figure and reduction

Publication figure: [Seven-row control diagnostic](../figures/03-control-coherence-448-800.pdf).
Generator: [04_control_coherence_figure.py](../04_control_coherence_figure.py).
Plotted values: [control_coherence_448_800.csv](../control_coherence_448_800.csv).
Audit: [control_coherence_verification.json](../control_coherence_verification.json).

At completed optimizer state $t$, let $g_{i,t}$ be the gradient of one example's
half-squared-error loss, including every weight and bias, and let
$g_t=N^{-1}\sum_i g_{i,t}$ be the full-objective raw gradient. $N=5000$.
The seven rows, including the denominator and output-change panels, are:

1. Incoming loss change, $L_t-L_{t-1}$. This is the signed difference in full
   training objective, not the loss itself or a relative/percentage change.
2. $\eta\widehat\kappa_t=1-\rho_t\|g_t\|/\|g_{t-1}\|$. The notation matches
   the existing overview figures. Under Adam it is normalized projected
   gradient change, not a measured curvature or general stability threshold.
3. $\rho_t=\cos(g_t,g_{t-1})$: temporal alignment of raw full-batch gradients.
4. $\|g_t\|$: norm of the mean raw gradient, before Adam's transformation;
   this is the numerator of C when written as a ratio of means.
5. $N^{-1}\sum_i\|g_{i,t}\|$: mean individual-gradient norm, the denominator.
6. The user-defined intra-batch coherence, explicitly labeled C = panel (d) /
   panel (e):

$$
C_t=\frac{\left\|\sum_{i=1}^N g_{i,t}\right\|}
{\sum_{i=1}^N\|g_{i,t}\|}
=\frac{\|g_t\|}{N^{-1}\sum_{i=1}^N\|g_{i,t}\|}.
$$

7. Consecutive output-change alignment, where $F_t$ concatenates all10 raw
   outputs for all5000 fixed examples in the same order:

$$
\Delta F_t=F_{t+1}-F_t,\qquad
\rho_t^{(\Delta F)}=\cos(\Delta F_t,\Delta F_{t-1}).
$$

Panel(g) at $t$ compares the incoming prediction change $t-1\to t$ with the
outgoing change $t\to t+1$. This is the convention used in the earlier output
alignment plot. It differs from panel(c)'s raw-gradient pair at states$t-1,t$;
panel(a) is the loss change ending at$t$. Panel(g) requires output states447
through801 to cover448..800. It is a single cosine of two50,000-coordinate
vectors, not an average sample-wise cosine, output/target alignment, or cosine
of the predictions themselves. Raw outputs are FP32; differences and reductions
use FP64. Zero displacement gives null; all353 plotted pairs are defined.

This $C_t$ is a norm ratio, distinct from the earlier squared-energy ratio
$C_{\mathrm{energy},t}=\|g_t\|^2/(N^{-1}\sum_i\|g_{i,t}\|^2)$ used in
Figures01/02 and observations001/002. It is not obtained by simply taking the
square root of that earlier statistic. The mean of individual gradient norms
is required. It is also not the average of normalized pairwise cosines.

For each example, sum squared weight/bias gradient contributions across all
layers first, take the square root to obtain the whole-model norm, and then
average across examples. Each linear-layer weight contribution is
$\|\delta_i\|^2\|a_i\|^2$, and its bias contribution is $\|\delta_i\|^2$.
No parameter groups or examples are omitted. The full gradient numerator comes
from native autograd. A zero mean individual norm gives an undefined/null C;
all plotted denominators are nonzero. All temporal denominators are also nonzero.

## Caption

**Unchanged Adam control, CIFAR-10 MLP, states448--800.** Seven vertically stacked
panels show signed full-objective loss change, normalized projected gradient
change, adjacent raw-gradient cosine, norm of the mean raw gradient, mean
individual-gradient norm, and norm-based
intra-batch gradient coherence, followed by consecutive output-change alignment.
Every row includes all353 integer states,
including both endpoints, without smoothing or subsampling. State447 provides
the incoming history for the first plotted transition. Dashed references mark
zero loss change, projected change one, and zero gradient/output-change cosines. The projected-change
reference is a descriptive algebraic boundary, not an Adam stability guarantee.
Colors distinguish diagnostics; all panels share the completed-update axis.
Panel (e) is reduced from the already retained per-example energies; adding it
requires no new experiment. The titles and footer show that panel (f) equals
panel (d) divided by panel (e), with $g_t=N^{-1}\sum_i g_{i,t}$. All353 panel
ratios are verified, and the original five data series are unchanged.

## Matched experiment and verification

Same seed0, fixed5000 CIFAR10 training examples, 3072-200-200-10 tanh MLP,
physical batch1000, all656810 parameters, Adam LR.001, betas(.9,.999), eps1e-8,
no decay or clipping. Full half-squared-error objective; diagnostic vectors
FP32, reductions FP64, TF32 disabled. No new validation or test measurements.

Run238 replays0..800 so model/optimizer history remains identical. All801
loss/raw-gradient-norm pairs match Run223 exactly; all800 defined adjacent
cosines/projected changes match within1e-12. Full model/optimizer/RNG/current
and previous gradient equality passes at512 and800. The observer preserves
RNG and native parameter gradients. Frozen source and reference hashes are
recorded in the run's input manifest.

The retained `example_energies.pt` contains all1,765,000 per-example squared
norms (353 states x5000 examples). Independent NumPy square-root/sum reductions
reproduce the requested C to1.12e-16 maximum absolute difference. All88 states
overlapping Run237 reproduce its mean per-example squared-gradient energies
within4.45e-16 relative difference. Small double-precision autograd tests verify
the outer-product formulas, bias inclusion, full-example norm reduction,
non-divisible physical batches and zero-denominator behavior.

For Run238, all43 input hashes and10 downloaded artifact hashes pass. All2,471 plotted
values are checked against the independent reductions and original control.
The vector PDF was rendered and visually reviewed for legibility and clipping.
The Spark replay completed in10.113s, with peak allocated GPU memory286,736,896
bytes. No cloud resources or costs; worker exit and retained storage are recorded
in [Run238 completion](../../../runs/238-2026-11-07-mlp-intrabatch-coherence/completion.json).

Run239 captures355 raw output arrays447..801. All801 reference loss/norm pairs
through800 match exactly, and full states512/800 match. State801 is a declared
one-update continuation; a separate CPU reconstruction from the authenticated
state800 reproduces its parameters within7.46e-9 maximum absolute difference.
Its previous gradient and RNG match the parent exactly. All353 output cosines
are independently reduced with NumPy. All88 overlapping FP32 residual arrays
match Run237 exactly; the86 overlapping cosine values differ by at most2.16e-10
when using raw-output differences instead of rounded FP32 residual differences.
All355 output arrays reproduce the native full losses within3.30e-9 absolute
error. Run239's43 input hashes and10 retrieved artifact hashes pass. It completes
801 updates in12.297s with241,174,016 peak allocated GPU bytes, zero cloud cost.
Retained storage and worker exit are recorded in
[Run239 completion](../../../runs/239-2026-11-07-mlp-output-change-alignment/completion.json).

## Sources, coverage and limits

Sources are Run238's retained `artifacts/spark/payload/worker/coherence.jsonl`,
`boundaries.jsonl`, and `example_energies.pt`, with exact references from
Run223's prefix/control boundary histories and end checkpoints. The generator
and verification record include source, plotted CSV and PDF hashes.
The final row additionally uses Run239's `outputs.pt`, `output_alignment.jsonl`,
`boundaries.jsonl` and `end.pt`; their hashes and coverage are recorded separately
under `output_alignment` in the figure verification record. All six prior
data series remain unchanged when the seventh row is added.

The window's requested norm-based C spans about.11956--.37103. These values
must not be compared numerically with the earlier energy coherence as if the
definitions were equal. The plot is descriptive evidence for one seed and
trajectory. No confidence interval over seeds is available; numerical agreement
does not measure scientific uncertainty. Coincident peaks do not establish
that intra-batch alignment initiates the loss oscillation.
