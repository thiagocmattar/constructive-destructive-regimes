# Run222: matched Spark fixed-step interventions

Completed all 36 branches without failure: ten primary conditions and two optional
cosine durations across the three highest-LR model/dataset cases. Each starts
from exact transferred warm state 1024 and performs 64 full-objective updates to
1088, with warm Adam moments and no prefix retraining. MLP uses LR 0.001;
ResNet-32 and Transformer use 0.003. The optional durations were admitted only
by the frozen worker's time rule.

[Analysis043 findings](../../analyses/043-2026-11-03-spark-fixed-step-interventions/FINDINGS.md),
[all 13 response PDFs](../../analyses/043-2026-11-03-spark-fixed-step-interventions/REPORT.md),
[final retrieval receipt](retrieval.json), [completion record](completion.json),
and [verification](../../analyses/043-2026-11-03-spark-fixed-step-interventions/verification.json).

The frozen [config](config.json) tests unchanged highest LR, fixed half/quarter,
cosine32 floors 0.5/0.25/0.1, global raw-gradient clipping at 1x/2x historical
median and decoupled decay 0.01/0.1. Optional cosine durations 8/64 use floor
0.25. The [Run221 protocol](../221-2026-11-03-spark-fixed-step-interventions/PROTOCOL.md)
pins literature motivation, prepared data, objectives and optimizer semantics.
Fifteen focused mathematical/native-optimizer checks passed before launch.
No executed scientific source, config, horizon, seed or variant was changed.

Run221's transfer qualification is retained: ResNet relative gradient difference
0.0087187473 exceeds the frozen 0.001 gate despite exact model/moment tensors and
loss relative error 2.78e-7. Run222 uses fresh matched Spark controls and requires
exact same-parent observations plus two direct versus one-plus-one restored
weights/moments/previous raw gradient/RNG. All three cases pass, and the downloaded
checkpoints independently reproduce those equalities. Historical bitwise replay
is not claimed; the before/after hardware handover is explicit in every main figure.

The worker launched at 2026-11-03 22:52 UTC and finished at 2026-11-04 09:52:26
UTC (06:52 BRT), before the 07:45 BRT cutoff. Run222 records 2,304 main updates,
12 qualification updates and 2,361 full-gradient calls. All 347 transferred files
cover the 337-file terminal inventory and pass size/SHA256 checks. All 31 frozen
inputs, 19 executed source copies and 45 endpoint checkpoints are verified.
Raw evidence remains under artifacts/snapshots, selected by retrieval.json.

The final local payload plus archive occupies about 10.66 GB; the isolated Spark
task directory retains about 7.31 GB allocated. The worker PID/PGID2439138 has
exited; unrelated suspended PID458832 is preserved. Owned Spark, USD 0 cloud
spending, no manuscript edit or push. The follow-up monitor record is in
[monitor.json](monitor.json).

Collection and reproduction from the repository root:

    python runs/222-2026-11-03-spark-matched-interventions/03_collect.py --final
    python analyses/043-2026-11-03-spark-fixed-step-interventions/01_report.py
    python analyses/043-2026-11-03-spark-fixed-step-interventions/02_verify_final.py

The collector supports Windows long paths and verifies archive/per-file hashes.
Use --final only after terminal manifest and inventory. One selected seed/state
and a 64-update horizon support short response findings, not universal stability,
long-term convergence, optimal hyperparameters or held-out benefit claims.
