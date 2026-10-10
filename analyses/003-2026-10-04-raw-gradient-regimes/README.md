# Analysis 003: Model size, loss, and adjacent raw-gradient cancellation

Complete local analysis of the pilot runs, extended with the now-completed
three-seed replication. The scientific reductions made no cloud calls and did
not train or change run evidence. A subsequent read-only Runpod catalog quote
is retained for the ETC question; Analysis004 owns the new execution. This
numbered folder owns the raw-gradient/loss and observational LR questions;
Analysis002 owns the original replication evidence.

The larger models have lower training and validation losses, more frequent
negative adjacent raw-gradient cosines, and greater average negative contribution
over all pairs. Among negative pairs alone, their mean magnitude decreases
slightly. Thus the proposed dimension-mediated explanation, "larger models learn
better because their regime becomes less destructive," is not established.
The meaning of intensity and the measurement horizon change the conclusion.
All three paired seeds retain these directions; the table below is the original
seed1234 pilot. See the [replication audit](observations/003-replicated-conclusions.md)
for every seed and the [recommended schedule experiment](SCHEDULE_EXPERIMENT.md)
for a focused follow-up.

| Model | Negative pairs | Mean max(-cosine, 0) | Mean magnitude among negative pairs | Final validation loss |
| --- | ---: | ---: | ---: | ---: |
| 14M | 350/711 (49.23%) | 0.160561 | 0.326168 | 5.182392 |
| 31M | 405/711 (56.96%) | 0.175342 | 0.307822 | 4.611768 |
| 70M | 481/711 (67.65%) | 0.200192 | 0.295919 | 4.063201 |

Read the [conclusion and manuscript assessment](REPORT.md) and
[scientific observation with contracts and captions](observations/001-raw-gradient-regimes.md).
The [requested 1x2 panel](figures/loss_and_raw_cosine.pdf) compares training loss
and raw cosine. A [second 1x2 panel](figures/negative_alignment.pdf) separates
negative-pair frequency and unconditional intensity. Publication figures are PDF.

## Comparison and hypothesis

H003 records the user's direction: increasing model size reduces the negativity
or intensity of the destructive regime, potentially through dimensionality and
thereby explaining better performance. Here "destructive" is operationalized as
local raw-gradient cancellation (cosine < 0), without presuming harm to learning.
Frequency, unconditional intensity, and conditional magnitude are reported
separately. Evidence is mixed across definitions; the proposed causal explanation
is untested. This question was posed after the pilot existed and is exploratory.

Primary cohort: Run 002 attempt 02 (14M), Run 003 attempt 01 (31M), Run 004 attempt 01
(70M), all seed 1234. Run 001 attempt 04 is a same-seed 14M replay sensitivity,
not an additional independent seed. Initialization recipe, token cache/order,
712 updates, batch, loss reduction, AdamW settings, schedule, precision, and
full validation coverage match in the primary cohort. Width/head configuration,
parameter shapes/values, and width-dependent initialization scales vary with size.

## Reproduction and verification

From the repository root:

```powershell
python analyses/003-2026-10-04-raw-gradient-regimes/01_analyze.py
python analyses/003-2026-10-04-raw-gradient-regimes/02_verify.py
```

Dependencies: NumPy, Matplotlib, and PyMuPDF. The first script uses Analysis 001's
read-only matched-cohort loader and checks completed-source hashes against its
frozen results; it writes exclusively inside this folder. Neither script reads
the active runs' event streams, changes their configuration, or calls Runpod.
`protected_inputs.json` fingerprints manuscript files and the live runs' root
scientific inputs; verification rechecks them without writing to either location.

`results.json` contains definitions, full and display-only summaries, fixed
quarters, 15/31/61-update smoothing sensitivity, source/config hashes, and the
Run 001 replay check. `figure_data.csv` retains all 2,136 update records, undefined
first cosines, raw values, and the plotted reductions. `verification.json` records
five mathematical checks, source/preservation checks, and figure hashes/bounds.
All three PDF renders were visually inspected. The original figures describe
seed1234; the replication audit reports three seeds separately. Correlated steps
are never treated as independent replications.

## Extension: schedule and late recovery (H004)

The [schedule observation](observations/002-schedule-and-recovery.md) and
[1x2 schedule/alignment panel](figures/schedule_and_raw_cosine.pdf) investigate
whether LR decay contributes to late positive recovery. The timing is compatible,
but LR and training progress are confounded. All models have an identical schedule;
the 70M trajectory returns to negative alignment even after LR reaches its fixed
floor. Earlier recovery therefore affects the global size summaries without
identifying a schedule cause or a dimension-only mechanism.

Reproduce with `python analyses/003-2026-10-04-raw-gradient-regimes/03_schedule.py`,
then rerun `02_verify.py` for preservation and PDF checks. `schedule_data.csv`
retains all 2,136 boundary records, including current/preceding LR. The complete
schedule identity, frozen-code reconstruction, phase summaries and smoothing
sensitivity are in `schedule_results.json`. Three additional mathematics checks
cover LR scaling invariance, sign recovery on a contracting quadratic, and the
descriptive recovery criterion. No new training or schedule intervention was run.

## Extension using the completed seed replication

Reproduce with `python analyses/003-2026-10-04-raw-gradient-regimes/04_replication_check.py`.
The script verifies Analysis002's reduction hashes, 63 source/receipt hashes,
agreement with its reported statistics, D = F*M, and the 53 protected inputs.
It writes only `replication_check.json` in this folder. Observation003 preserves
the restricted conclusion: larger models have lower validation loss alongside
more frequent local cancellation, and net whole-trajectory interaction stays
positive in all nine runs. Conditional negative magnitude decreases in 3/3 seeds.

The proposed next experiment branches existing update356 checkpoints for 14M
and 70M, across all three seeds, into original, held, and accelerated decay.
It comprises 18 continuations and tests the contribution of late decay,
conditional on a common trained prefix. Resume mechanics require implementation
and verification; adaptive schedules are deferred. No new run was launched.

The authorized first single-seed schedule test is now complete in [Analysis004](../004-2026-10-04-lr-schedule-intervention/REPORT.md). Its schedule-dependent raw cancellation/loss results are separate from, and preserve, the original-recipe replicated conclusions above. The live planning quote is retained in [runpod_price_snapshot.json](runpod_price_snapshot.json); the new run costs and cleanup belong to Analysis004.
