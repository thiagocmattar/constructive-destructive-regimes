# Run 267: uniform neuron q75

H047 / ongoing Analysis065, I23. First-layer policy `uniform` from the
untreated full state448; first changed update448->449, end801. No inherited
freezing or LR reduction. All native Adam LRs remain0.001. All raw gradients,
moments and clocks advance; only first-layer weights/biases are transformed
after native Adam. Later-layer parameters remain native on this trajectory.
Threshold key `q75` is fixed by Run258 using only updates416-447. Criterion3
is per-neuron preactivation displacement RMS over all5000 fixed examples.
For uniform controls the paired source is `runs/262-2026-11-07-mlp-gate-neuron-q75`; per-step first-layer norm
matches that source within measured FP32 rounding, with factor>1 allowed.

Primary metrics: mean loss448-800 and excursion sizes; event450-500, last32
mean769-800, endpoint800, ten consecutive states below0.02, representation
cosines/RMS, applied/removed increment norms and selected fractions. Controls
and all treatments compare with unchanged Adam; single-seed/fork inference.
Dense states447-501 and gate audits448/474/800 support independent reductions;
all-state summaries and full recovery checkpoints retained. Zero-motion factors
are1 for neuron cap and0 for zero-target uniform; no division epsilon.

Owned Spark, no cloud charges. Prior continuations8-10s; extra FP64 sample-space
products are uncalibrated, allow2min/run plus retrieval.600s worker and630s
detached timeout with30s grace; all10 branches max110min including kill grace.
At most30GB total retained including archives. Progress every64 updates;
STOP/signal/time/nonfinite guards. Source identities and size/SHA256 retrieval
must pass before closeout; preserve unrelated work and all earlier figures.

## Completed result

Mean loss 448-800: 0.02142387 (-6.07% versus unchanged
Adam); maximum 0.03478547 at 450; tail mean 769-800:
0.01498385. Runtime 35.07 s; completed
state 801. All 73 input identities, 26 retrieved files,
full-state qualification, independent gate/dense checks and recovery clocks pass.
Worker exited; raw evidence and recovery states retained locally and on owned
Spark, no cloud charge. [Comparison and limitations](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/006-selective-update-gates.md).
