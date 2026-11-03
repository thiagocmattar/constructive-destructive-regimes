# Run221: Spark calibration and transfer qualification

The cross-GPU gradient gate failed for ResNet before any treatment production.
The original failed qualification is retained. [Run222](../222-2026-11-03-spark-matched-interventions/README.md)
continues as a matched same-Spark comparison with an explicit design amendment.

H029 / Analysis043. User authorized overnight tests of literature-motivated
interventions at the highest tested LR for all three existing model/dataset cases.
[Protocol](PROTOCOL.md) and [config](config.json) fix the design. Ten primary
conditions x three cases x64 updates start from authenticated warm state1024;
two additional cosine durations are conditional on time. Highest LRs are0.001
for MLP/CIFAR-10 and0.003 for ResNet-32/CIFAR-10 and Transformer/WikiText-2.
All branches retain native Adam state and full original objectives/partitions.

Spark GB10 only, zero cloud cost. Production-shaped timing is about0.017/15/35s
per complete MLP/ResNet/Transformer update. The earlier512+512 planning sketch
was superseded before production to fit the overnight envelope. Compute stops
by2026-11-04 07:45 BRT; report target08:00 BRT. Fifteen focused CPU checks pass.
Warm-parent and exact within-Spark resume qualifications gate production.

01_worker.py owns frozen scientific execution;02_qualify_parents.py checks warm
weights/moments and the original-to-Spark gradient bridge without any updates.
The first mechanics profile fails on seed metadata indexing before an update;
00_profile_retry.py correctly reads its list and executes9 mechanics updates.
All profile sources/logs are retained. Detached status/events/recovery prevent
loss of provenance on SSH interruption. Old queues and shared data are preserved.

[Analysis043](../../analyses/043-2026-11-03-spark-fixed-step-interventions/README.md)
owns before/after figures, numeric comparisons and eventual findings. Historical
preperiod and fresh Spark baseline are explicitly distinguished. No manuscript
edit, cloud provisioning or push is included.
