# Analysis062: what amplifies the MLP loss spikes?

The [causal intervention tests remain TODO](TODO.md) while
[Analysis063](../063-2026-11-07-architecture-instability-tracing/README.md)
formalizes and applies the tracing method to ResNet and the Transformer.

[Input distributions and intervention options](observations/013-input-distributions-and-intervention-options.md)
verify that all3072 coordinates are already individually standardized: actual
5000-example column stds0.98475--1.02076. Strong correlations remain (first input
covariance mode26.85%, first ten64.50%). Selective first-layer LR/freeze and
function-preserving covariance interventions are proposed, not executed. Static
local audit only; script18, per-column CSV and source-hashed results retained.

Run237 / H042 follows the RMSNorm question with an exact Adam-control replay.
The actual MLP has no activation normalization. Test the analogous concern in
Adam's denominator, separate per-example gradient strength from cancellation,
and measure network sensitivity and actual-step loss curvature. The full 5,000
training examples are used at every state 448..535; seed 0 only. No new training
policy or manuscript edits. See the owning run for the frozen measurement plan.

After observing almost constant mean Jacobian traces but changing sensitivity
along the actual updates, `02_cross_jacobian_probe.py` is a declared post hoc
static probe. It crosses baseline/incoming-peak Jacobians with baseline/incoming
Adam increments. This distinguishes a changed update direction from a changed
network Jacobian in those directions; it does not estimate the largest singular
value or establish a universal stability threshold. Both baseline and incoming
directions are reconstructed from retained exact model/optimizer/gradient states.
All examples are included. No additional trajectory is trained. Expected seconds
on the owned Spark, zero cloud cost, five-minute cap, small JSON output retained.

Post hoc residual projections and class/example/neuron summaries are descriptive
reductions of the retained full arrays. A direction fitted to a selected event
is explicitly labeled as such; correlations are not independent causal tests.

## Completed findings

The large peaks show nearly reversing prediction increments (cosines about
-.985 and -.990), broader example-wise loss increases, and reduced cancellation
between individual gradients. The batch gradient grows 4.13x/5.33x while
per-example gradient RMS grows 1.63x/1.88x. Mean parameter-Jacobian RMS changes
only .50%/.36%. Crossing earlier/later Jacobians with the actual directions
shows the late directions were already more sensitive in the earlier network.
Adam increments grow; denominators do not collapse. Tanh attenuates the measured
backward vectors. Exact output-space loss terms quantify the finite-step
overshoot; all 87 actual increments start downhill, but 42 increase loss.

- [Full observation, definitions, tables, captions and limits](observations/001-gradient-amplification.md)
- [Definitions and equations: coherence, prediction reversals and dispersion](observations/002-coherence-prediction-changes-and-dispersion.md)
- [Seven-row control figure, states448--800, with coherence and output-change alignment](figures/03-control-coherence-448-800.pdf)
- [Temporal overlay and scatter: gradient versus output-change alignment](figures/04-gradient-output-alignment-comparison.pdf)
- [Loss, alignment overlay and fixed SVD output-change patterns](figures/05-network-output-heatmap.pdf)
- [SVD heatmap definitions, energy coverage and caption](observations/006-svd-output-change-heatmap.md)
- [Leading SVD coefficient and spike zooms](figures/06-leading-svd-coefficient.pdf)
- [Projected reversals: definitions, counts and verification](observations/007-leading-svd-coefficient.md)
- [Didactic guide: output oscillations, SVD dimensions, coefficients and hidden representations](observations/008-understanding-output-oscillations-svd-and-hidden-representations.md)
- [Raw-gradient, output-change and hidden-change SVD comparison](figures/07-three-space-svd-comparison.pdf)
- [Visible/null hidden motion and output attribution](figures/08-hidden-visible-nullspace.pdf)
- [Three-space results: concentrated motion with an oscillating null component](observations/010-gradient-output-hidden-svd-results.md)
- [Corrected contract: raw gradients, not gradient differences](observations/009-three-space-svd-contract.md)
- [First-hidden, preactivation, second-hidden and output SVDs](figures/09-first-hidden-svd-comparison.pdf)
- [Transmission through the second linear layer and tanh](figures/10-first-to-second-hidden-transmission.pdf)
- [Earlier-layer results: oscillation is already present in the first hidden representation](observations/011-first-hidden-oscillation.md)
- [Complete forward-layer SVD comparison, including the fixed-input check](figures/11-all-layer-svd-comparison.pdf)
- [First-layer actual steps, first tanh and fixed input](figures/12-fixed-input-first-layer-transmission.pdf)
- [All-layer conclusion: static inputs, oscillating learned projections](observations/012-all-layers-and-fixed-input.md)
- [Historical mean-output heatmap definitions](observations/005-mean-output-heatmap.md)
- [Alignment comparison definitions, statistics and caption](observations/004-gradient-output-alignment-comparison.md)
- [Extended-window definitions, sources and verification](observations/003-control-coherence-448-800.md)
- [Complete diagnostic time course](figures/01-gradient-amplification.pdf)
- [Oscillations and crossed-direction probes](figures/02-oscillation-and-direction-probes.pdf)
- [Numerical reductions and source hashes](results.json)
- [Independent verification](verification.json)

Reproduce using retained artifacts:

```powershell
python analyses/062-2026-11-07-mlp-gradient-amplification/02_cross_jacobian_probe.py --device cpu
python analyses/062-2026-11-07-mlp-gradient-amplification/03_verify.py
python analyses/062-2026-11-07-mlp-gradient-amplification/01_report.py
```

Both vector PDFs were rendered and visually reviewed. All 536 reference pairs,
15 anchors and final535 state pass; 57 input and 28 artifact hashes verified.
All 10 crossed probes agree between CPU and GPU within1.01e-7 relative error.
Worker10.458s, static Spark probe1.278s; workers exited. About391MB retained on
owned Spark including archive, no cloud resources or charges. No manuscript
changes. One selected seed/window; no unique initiating cause established.

Run238 subsequently extends measurement through800 for the single-column figure:
loss change, eta*kappa_hat, rho, gradient norm, mean individual-gradient norm and
C=||sum_i g_i||/sum_i||g_i||. This new C is distinct from the squared-energy
coherence in the preceding analysis. All801 reference pairs and full states512
and800 match. All353 requested states include all5000 examples;2,118 plotted
values pass independent checks. The added panel(e) uses retained data; panel(f)
explicitly equals panel(d) / panel(e). Reproduce with
`python analyses/062-2026-11-07-mlp-gradient-amplification/04_control_coherence_figure.py`.

Run239 adds panel(g), cos(F_(t+1)-F_t,F_t-F_(t-1)), from complete raw output
arrays447..801. The same control is exactly replayed through800, then advanced
one unchanged Adam step for the endpoint. All353 output cosines are verified;
all six prior curves remain unchanged. The current PDF contains2,471 values
across seven rows. Full provenance and indexing are in observation003.
