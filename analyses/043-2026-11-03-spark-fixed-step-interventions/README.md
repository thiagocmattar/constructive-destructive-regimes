# Analysis043: fixed-step intervention responses on Spark

Run222 matched Spark production after retained Run221 transfer qualification; [Run221 protocol](../../runs/221-2026-11-03-spark-fixed-step-interventions/PROTOCOL.md).
Fixed step1024,64 new updates, highest tested LR for MLP/CIFAR-10,
ResNet-32/CIFAR-10 and Transformer/WikiText-2. Ten primary conditions cover
unchanged, fixed LR reductions, cosine floors, raw-gradient clipping and decoupled
weight decay. Two additional cosine durations require enough remaining time.

The figures will compare each intervention with unchanged in three model columns,
with solid lines, independent panel ranges and raw rho restored. A marked boundary
separates shared historical preperiod and fresh matched Spark continuations.
Loss, rho, gradient norm, projected raw-gradient change and LR are primary rows;
parameter/actual-increment norms accompany weight decay. Source values and
qualifications remain inspectable. Single selected seed/state, short response
window; no long-term convergence or held-out gain is assumed.

Run221's ResNet cross-GPU gradient gate failed; exact tensors and near-identical
loss are retained. Run222 requires exact within-Spark replay and uses fresh Spark
controls. Historical preperiod is context with an explicit hardware boundary.
