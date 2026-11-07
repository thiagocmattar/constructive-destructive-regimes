# Run 246: I3, frozen linear1 with Adam LR0.02 from500

Same continuing MLP analysis064/065 and H045. Compare this complete intervention
against unchanged Adam and display beside I1 and I2. Freeze first-linear weights
and biases from448; set LR0.02 at500 (10x I2, 20x baseline). First changed update
is500->501, end801. No optimizer reset, unfreeze or other setting changes.
Same seed0, all5000 examples/order, full mean half-squared-error objective,
physical1000, FP32/TF32off and native Adam(.9,.999), eps1e-8, no clipping/decay.
Retain full gradient/representation captures499..551, rolling diagnostics500..800,
actual increments and complete fork500/qualification501/recovery/final states.
Reconstruct the same frozen parent from448; require exact qualification through501.
First-step moments/clocks must match native and increments scale20x within FP32
rounding (absolute tolerance2e-6 because the low-rate rounded increment is scaled).

Owned Spark, no cloud resources or charges. Run245 measured7.76s; qualified ETC
under2min including I/O,600s worker limit, detached630s timeout and30s TERM grace.
Retain below3GB remote including archive, verify66 inherited/new sources plus
shared continuation helper, verify all retrieved sizes/SHA256 before closeout.
Monitor persisted progress/logs; stop on nonfinite loss/gradients and preserve
failure evidence. Do not stop solely because finite loss rises. Preserve unrelated
workloads. One seed/fork/horizon; training behavior only, no generalization claim.

## Completed result

Run completed state801 in7.960s (301 intervention updates in6.637s), peak allocated
GPU memory456.7MB. All67 source files and22 retrieved files verified. Exact parent
qualification, LR-only start-state change, moments/clocks, first-layer constancy
and finite final parameters pass. First-step scaling error1.49e-7. No task worker
remains;1.911GB retained on owned Spark, verified copies local, zero cloud cost.

LR0.02 produces severe overshoot: loss107.099 at504, recovering to0.308132 at800
versus0.0302502 for unchanged Adam. First-layer weights/biases remain fixed.
The same ongoing [Analysis065](../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md)
now compares I1/I2/I3 side by side, each against unchanged baseline; see its
[I3 observation](../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/002-frozen-lr-0p02.md).
