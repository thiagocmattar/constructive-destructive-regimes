# Observation 001: paired fixed-CE curvature experiment

Status: complete and verified, October 23,2026. H022; scientific Runs 155-184,
with Run 154 calibration excluded. See the [conclusions](../REPORT.md).

## Sources and reduction

The [cohort](../cohort.json) names all ten paired seeds and three CE arithmetic
conditions. Each owning run retains 10,000-update `metrics.jsonl.gz`,23 regular
curvature states, five first-loss crossings, sparse next states, recovery
state and byte/SHA256 inventories. All 300 k scientific updates,840 ordinary and
660 Adam-scaled eigenpairs pass [verification](../verification.json).
The [input inventory](../source_inventory.json) freezes source/config/initial
tensor identities. No raw evidence is rewritten by reduction.

`08_analyze.py` verifies, reduces the five predeclared late states per seed,
calculates paired t intervals in log space, enumerates sign-flip sensitivity,
and prepares [figure data](../figure_data.json). `09_plot.py` creates the
[publication PDF](../figures/fixed_ce_curvature.pdf). The [figure manifest](../figure_manifest.json)
records source/data/PDF hashes. `10_checkpoint_audit.py` is a separately labeled
post-hoc same-state gradient audit; it performs no parameter updates.

## Observation

Fixed/native late maximum-curvature ratio is 0.0148854,95% CI[0.00307713,0.0720074],
with lower values in 10/10 paired seeds. The FP64-loss/native ratio is 0.0210346,
95% CI[0.00414967,0.106625]. Fixed/FP64-loss equivalence is not established:
seed 3 differs substantially and remains included.

At five matched-loss levels, fixed/native median curvature ratios span
0.9863-1.0877. All 100 candidate loss-matched pairs pass the 10% matching rule.
Late lambda/F fixed/native ratio is 0.8815, while late loss ratio is 0.01689.
These controls suggest that most ordinary-curvature reduction accompanies
lower attained loss. Adam-scaled curvature instead has fixed/native ratio 2.409.
The late mean raw cosine becomes more negative (-0.2585 to-0.6651), so these
diagnostics must not be substituted for one another or for formal stability.

None of the 10 native conditions meets the hard-collapse exposure rule (8 required).
The native arm's maximum observed fraction is 2/480, below the 10% exposure rule.
An exploratory audit of 50 fixed-arm late checkpoints finds native median
parameter-gradient relative error 0.1699 versus 1.38e-7 for fixed CE against a
FP64-loss-only reference. This is evidence of rounding before exact collapse,
not a qualified reproduction of the paper's hard-collapse mechanism.

## Caption

**Fixed cross-entropy and attained curvature under paired AdamW training.**
One width 64 Transformer solves modular addition mod 31 with 480 training and 481
held-out pairs. Ten initialization seeds share exact initial tensors across
native FP32 CE, fixed FP32 CE and a FP64-loss-only control; each runs 10,000
full-batch updates. (a) FP64 CE on FP32 logits every 20 pre-update states, with
terminal dots from the full-FP64 objective at state 10000. (b) Native target
gradient collapse, with its 10% qualification threshold. (c-f) Common full-FP64
training-Hessian maximum eigenvalue, gradient-direction curvature, actual-update
curvature and Adam-scaled maximum eigenvalue. Thin curves show seeds and thick
curves their medians. (g) Predeclared paired late-window ratios, with geometric
means and 95% paired log-space t intervals; dashed line marks 20% reduction.
(h) Paired ratios at first-crossing matched-loss levels; diamonds are medians.
Every Hessian covers all 480 training pairs and largest algebraic eigenpairs
have fresh relative residual<=0.005. Terminal actual-update curvature and
initial Adam-scaled curvature are undefined and omitted. All raw histories
are retained; the displayed spectral cadence cannot resolve every spike.

## Coverage, uncertainty and limits

Initialization is the uncertainty unit; checkpoints and repeated loss levels
within a seed are correlated. The split, task, architecture, optimizer,
precision and horizon are fixed. Ordinary curvature is coordinate dependent;
matched loss and lambda/F do not remove all rescaling effects. ReLU is piecewise
smooth, and finite-step Taylor remainders are retained. There is no global,
all-direction, convergence or Pythia-transfer claim. No manuscript edit.

The experiment used owned compute, zero GPU rental and no cloud resources.
All calibration/production evidence totals 1,061,488,148 bytes locally. The
PDF was rendered with Poppler, checked for label/legend/caption overlap,
iterated and visually accepted. Output/source hashes and documentation links
are checked in [output verification](../output_verification.json).
