# Run 245: double LR while the first MLP layer stays frozen

User-authorized H045 / Analysis065. Retain Run244's persistent first-linear
weight/bias freeze at state448, its data and Adam state. At state500, double
LR0.001 to0.002 for update500->501 and every subsequent update through801.
There is no optimizer reset or unfreezing. Nominal LR also changes in the
suppressed first block's original parameter group, but its actual increment
remains exactly zero; LR does not enter Adam moment updates.

Reuse the original5000 CIFAR-10 examples/order, half squared-error objective,
physical1000, tanh MLP3072-200-200-10, beta(0.9,0.999), eps1e-8, no clipping/decay,
FP32 and TF32off. Full raw gradients retain all656810 derivatives; updated-layer
metrics restrict to42210 parameters. Raw gradients are not differenced.

Load authenticated Run244 fork448 and replay its frozen policy to501. Require
exact loss/norm/adjacent metrics and all raw gradient tensors448-501; require
all five representation tensors499-501 to match retained evidence. Save the
reconstructed complete state500, restore it, then change only LR. Compare the
first doubled-rate Adam moment tensors/clocks to the native501 control; allow
only expected FP32 rounding when checking doubled actual increments.

Retain dense gradients/representations499-551, rolling all-layer/full-data
diagnostics through801, actual increments and full fork/qualification/recovery/
final checkpoints. Check first parameters and Z1/H1 for exact constancy, native
moment clocks, input order, observer RNG/gradient invariance and finiteness.
Primary outgoing interval500-550; complete intervention500-800; context448-499
comes from the unchanged retained parent. No new complete control is needed.

Owned Spark, zero cloud cost. Prior worker14.8s; expect under2min with I/O and
qualification,600s internal cap, detached630s timeout plus30s TERM grace.
Approximately1.1GB raw evidence; below3GB including archive. Persistent logs,
manifest/progress, source hashes, SHA256/size-verified retrieval and task-process
exit check. Preserve unrelated workloads. One seed/fork, training behavior only;
no inference of a universal stable LR or generalization improvement.

## Completed and verified

Completed state 801: 301 doubled-rate updates, 53 dense states. Worker runtime
7.762 s, intervention runtime 6.456 s, peak allocated GPU memory 456.7 MB.
All 66 source identities and 22 downloaded files pass verification. Exact parent
qualification, unchanged starting moments/clocks, zero first-layer increments,
finite final parameters and first-step doubled-increment checks pass. No task
worker remains; 2.021 GB retained on owned Spark, verified copies local, no cloud
cost. See `verification.json`, `retrieval.json` and `completion.json`.

This is I2 in the **same MLP analysis as Run 244 / I1**. Its complete intervention
is freeze first-linear weights/biases from 448 **plus** double LR from 500.
The primary comparison is with unchanged Adam, not with I1. The revised
[analysis and figure](../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md)
show both interventions against baseline. Over 448-800, I2 reduces output motion
and loss excursions but has 20.58% higher mean loss than baseline. Relative to I1,
endpoint loss at 800 is 3.04% lower while small output reversals increase.
