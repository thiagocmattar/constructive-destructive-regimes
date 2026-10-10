# Replicated raw gradient conclusions

The completed [Analysis002](../../002-2026-10-04-scale-seed-replication/observations/001-seed-replication.md)
reinforces the restricted conclusion from this analysis. In all three paired
seed blocks, larger models have lower validation loss, more frequent negative
adjacent raw-gradient cosines, greater unconditional negative intensity, and
smaller magnitude conditional on a negative pair. Local cancellation also
coexists with positive net interaction over the complete gradient sequence in
all nine trajectories. These observations do not identify dimensionality or
alignment as a cause of better performance.

## Sources and reduction

Generating script: [04_replication_check.py](../04_replication_check.py).
Machine-readable output: [replication_check.json](../replication_check.json).
The script reads Analysis002's completed `results.json` and `figure_data.csv`,
checks both against its generation hashes, and rechecks 54 run-source files and
nine acceptance receipts. Analysis002 and all run evidence remain unchanged.
The 53 manuscript and previously active run scientific inputs protected at the
start of Analysis003 retain their hashes.

Runs002/003/004 form seed1234; Runs005/006/007 form seed1235; Runs008/009/010 form
seed1236. For each size/seed, the reduction retains all 712 optimizer boundaries
and 711 defined adjacent cosines, including the final partial update. The raw
capture, accumulation, parameter scope, precision, dataset identities, optimizer,
and full token-weighted validation contracts are those of Analysis002.

For defined cosines rho, F is the fraction below zero; D is mean max(-rho, 0);
M is mean(-rho | rho < 0). The reduction verifies D = F * M and agreement with
Analysis002's published counts, median cosines, and terminal raw B/A. Steps are
weighted equally in these geometric summaries, while validation is weighted by
supervised tokens.

## All seeds retain the pilot ordering

| Seed | Model | Negative fraction F | Unconditional intensity D | Conditional magnitude M | Validation loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 1234 | 14M | 49.23% | 0.160561 | 0.326168 | 5.182392 |
| 1234 | 31M | 56.96% | 0.175342 | 0.307822 | 4.611768 |
| 1234 | 70M | 67.65% | 0.200192 | 0.295919 | 4.063201 |
| 1235 | 14M | 50.21% | 0.157530 | 0.313737 | 5.182016 |
| 1235 | 31M | 53.31% | 0.164349 | 0.308318 | 4.617403 |
| 1235 | 70M | 64.56% | 0.185030 | 0.286615 | 4.058427 |
| 1236 | 14M | 48.95% | 0.165687 | 0.338515 | 5.164867 |
| 1236 | 31M | 64.56% | 0.195395 | 0.302670 | 4.522797 |
| 1236 | 70M | 68.21% | 0.197547 | 0.289600 | 4.029232 |

All six strict orderings checked here replicate in 3/3 seed blocks: F and D
increase; M, raw median cosine, terminal raw B/A, and validation loss decrease.
This is consistent with Analysis002's predeclared primary replication results.
The additional D/M and whole-trajectory reductions are exploratory extensions,
not retrospectively predeclared endpoints of the replication runs.

Using E = sum ||g_t||^2 and B = ||sum g_t||, net raw all-pair interaction
J = B^2 - E is positive in all nine runs, ranging from +8047.20 to +11915.86.
Replacing it by norm-weighted lag-one interaction leaves a relative squared-norm
error between -98.57% and -97.90%. This extends the pilot's horizon distinction;
it is a raw-gradient identity, not an identity for actual AdamW parameter updates.

## Late differences also persist at the common LR floor

Among full-batch pairs576-711, preceded by LR = 0.0001, median raw cosine is:

| Seed | 14M | 31M | 70M |
| --- | ---: | ---: | ---: |
| 1234 | +0.360203 | +0.155987 | -0.003427 |
| 1235 | +0.340721 | +0.262725 | -0.022587 |
| 1236 | +0.304914 | +0.079358 | -0.021866 |

The fixed nominal LR does not erase size differences. This supports testing
schedule history and size-dependent responses; it cannot establish that decay
caused recovery. The descriptive schedule argument in observation002 remains
intact, with its timing and smoothing caveats.

## Interpretation limits

Three seeds test repeatability within this recipe. Initialization and data-order
seeds vary together; seed1234 used RTX4090 and the two new seeds used RTX5090.
Within-seed order is matched across sizes, but architecture and initialization
scales vary with size. The Run001 same-seed replay sensitivity still applies.
No seed confidence interval, scaling law, universal recovery threshold, two-phase
model, formal stability result, or causal performance mechanism is established.

The recommended next comparison is recorded in
[SCHEDULE_EXPERIMENT.md](../SCHEDULE_EXPERIMENT.md). It is a proposed intervention;
no schedule variants have been trained.
