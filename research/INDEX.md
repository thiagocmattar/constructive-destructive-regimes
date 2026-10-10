# Research index

## Current state

[Analysis012: Rosenbrock with weight decay](../analyses/012-2026-10-10-rosenbrock-adamw-lr/README.md)
completed Runs035-037: weight decay 0.1, LR 1e-2/0.003162277660/1e-3,
2,000 updates each. Final losses 0.001042468/0.000115599/0.234112; the
midpoint is lowest in this cohort. At 1e-2, decay raises final loss versus
the matched no-decay control despite mostly positive momentum cosine;
at 1e-3 it lowers final loss. All fail stationarity. The nine-panel PDF now uses
100-step cosine means and linear norm/A/B axes; all replay/optimizer/hash checks pass, including the Run030
prefix. Raw histories 2.91 MB retained locally; no cloud resources or billing.

[Analysis011: Rosenbrock learning rates](../analyses/011-2026-10-10-rosenbrock-lr/README.md)
completed the requested momentum/no-decay refinement: LR 1e-2/1e-3/1e-4,
2,000 updates each (Runs032-034). Final losses 0.000141348/0.325413/4.138634;
largest LR reaches the optimum neighborhood but continues oscillating, whereas
smaller LRs remain positively adjacent-aligned with less progress. All fail
the joint stationarity test. One large full-trajectory panel plus the exact
eight-panel diagnostic grid, including B/A ratios, is reviewed in a
single PDF. Replay/optimizer/geometry/hash checks pass; Run032 exactly matches
the Run028 prefix. Raw histories 2.85 MB retained locally; no cloud resources or billing.

[Analysis010: controlled Rosenbrock](../analyses/010-2026-10-10-rosenbrock-controlled/README.md)
completed four deterministic local CPU trajectories (Runs028-031),10,000 updates
each, varying momentum beta1=0/0.9 and decoupled decay=0/0.1 at matched
LR0.01/beta2=0.95/start(-1.2,1). All Analysis009 g/m/d metrics plus actual
displacements and exact temporal sums are retained. Every update/replay/hash
passed; eight PDFs and an offline linked step explorer passed visual/browser QA.
Final losses in run order are3.452e-6/0.02245/0.0009731/0.08264. None meets the
final100 sustained stationarity criterion. Lower loss with mostly negative
momentum cosine contrasts with the decay condition's positive late momentum
alignment; no alignment/convergence proxy or LLM transfer is established.
Raw histories17.13MB retained locally; no new cloud resources or billing.

[Analysis009 completed metric comparison](../analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/README.md)
adds four intermediate warmup-only14M LR points (Runs024-027). All four completed
712 updates/1.492B input tokens and full validation at18:36-18:37 UTC, with zero
overflows and all g/m/d reconstruction checks passing. Their36 compact result
files are retrieved/hashed; five seven-rate comparison PDFs passed visual review.
The combined raw-norm panel(e) is logarithmic; a new2x2 cumulative LR overlay
uses reversed plasma, with darker colors for larger LR. A new3x2 loss/raw-B/A/
alignment/norm view uses a50-update raw-cosine mean and10-update raw-norm mean.
A3x2 squared-geometry view compares A squared, B squared and their difference
for raw gradients and stored momentum, with independent logarithmic Y scales.
New validation losses are5.600857/5.237702/4.410370/4.318091 in increasing LR
order; LR0.00464159 is best among the four new hardware-matched conditions.
Its late momentum cosine0.852 is below0.956 atLR0.00215443, which has worse loss.
One-step alignment does not establish convergence/stability or long-lag effects.
The [manuscript interpretation](../analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/observations/004-implications-for-manuscript.md)
audits full-prefix raw all-pair interaction: positive at six LRs, negative only
at0.01 (also without the partial tail); the late sign change is not an early-warning result.
Frozen initialization/order/recipe match the original Spark conditions, but GPU
architecture/runtime differ across old/new runs. Full recovery downloads and
cloud cleanup are ongoing. Two owned two-GPU RTX4090 workers and60GB volume
mja1prpve4 remain; rateUSD3.56/h combined, deadline20:30 UTC, bounded estimated
compute/storageUSD12.69 within the approvedUSD15 cap. Independent guard and
compressed-batch retrieval controller are active; evidence retained until verified.

[Analysis008](../analyses/008-2026-10-08-no-decay-momentum/complete-three-lr/README.md)
completed all three Pythia-14M full passes on DGX Spark: warmup-only LR 1e-3,
1e-4 and 1e-2 (Runs021–023). Every run covered 712 updates and 1.492B input
tokens, passed full geometry/coverage verification and was retrieved with all
sizes/SHA-256 verified. Five final comparison PDFs, six per-run PDFs and the
automatic joint figure passed visual review; six interim PDFs are preserved.
Final validation loss is 4.764/6.232/6.636: baseline LR 1e-3 is best in this
matched single-seed cohort. LR 1e-2 has much larger raw norms and clipping,
raw B/A 0.0448 and worsening late training loss despite positive adjacent
momentum/adaptive cosines. Momentum lowers A substantially; its B stays close
to raw B at the two smaller LRs but differs at the largest, where clipping matters.
All runs share exact frozen initial tensors, order and runtime; LR scales decay.
Sparse checkpoint profiles do not recover missing dense short-lag histories.

The sequential queue finished at 2026-10-09 12:59:25 UTC (09:59 Brasilia),
after about 15 hours. Training ETC is zero; the 13:03 UTC check found GPU idle
at 38 C. All conditions had zero overflow retries. The local retrieval controller
finished automatically; recovery states and about 21.62 GB of retrieved outputs
remain locally and on Spark, with shared inputs also retained remotely. No cloud
rental or new experiment was launched. Scientific inputs remain frozen;
next IDs are Run028 / Analysis010 / H009. No stability or causal mediation claim.

[The completed Run021/022 interim analysis](../analyses/008-2026-10-08-no-decay-momentum/interim-021-022/README.md)
has the requested six-panel comparison and two coarse checkpoint temporal
supplements; all three PDFs are reviewed. At equal full-pass coverage, LR 1e-3
reaches lower validation loss (4.764 vs 6.232) despite more negative raw pairs
(84.53% vs 26.30%). Momentum remains positively adjacent-aligned in both.
Raw/momentum and adaptive B/A order the conditions differently; one seed does
not validate a geometric performance/stability proxy. Dense short-lag histories
are unavailable; the supplements explicitly use saved vectors at updates 100–700.

The [focused 2x2 view](../analyses/008-2026-10-08-no-decay-momentum/interim-021-022/observations/003-loss-alignment-norms.md)
marks the first negative raw pairs at updates 54 and 385. Raw-norm fluctuations
increase near those events, while momentum remains smoother. The norm gap is
already present before each first negative pair; cosine sign alone does not
define its onset. All six interim PDFs passed visual review.

[Run020](../runs/020-2026-10-08-dgx-spark-calibration/REPORT.md) verified SSH and
the user's GB10 DGX Spark runtime at 14M/31M/70M. Nine calibration updates project
approximately 5.0/5.7/7.6 hours per existing full-pass recipe (one seed); peak
allocated memory is 5.45–8.52 GiB. Compact evidence verified, GPU idle, 9.48 GB
task-owned local state retained at calibration closeout. Calibration only.
ARM-generated initialization hashes differed; Runs021–023 subsequently use a
common frozen checkpoint for identical starts.
[Analysis007](../analyses/007-2026-10-08-spark-runpod-timing/README.md) compares
that calibration with completed matched Runpod passes: Spark update times are
3.45–3.68× RTX4090 and 4.32–4.80× RTX5090. Cloud times cover full passes;
Spark times remain short-calibration projections. Analysis007 itself launched
no new run.

[Analysis006](../analyses/006-2026-10-07-cumulative-geometry-slopes/REPORT.md)
reanalyzes Run002 slopes: last 100 A/B increments remain unequal and both B/A
ratios rise. Raw B/A is nearly flat at 424-484 while B grows and training loss
declines. A flat ratio means proportional accumulation, not absence of learning.
Six PDFs, including 3x2 loss-difference grids at lags1/2/5/20, are verified;
sources preserved.

[Analysis005](../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) completed H005 at14M/31M/70M,seed1235: calibrated raw-gradient histories for all712 updates. Middle windows show negative lag1 and a positive longer-lag background, while exact all-pair interaction is positive in all nine windows. Full-pass lag1 truncation misses97.94-98.57% of squared cumulative raw-gradient norm. Early/late smooth decay to zero is not supported consistently over lags1-64. Six PDFs and two observations are complete; replays are numerically sensitive and do not add independent seeds. All six owned workers removed after verified retrieval; zero retained cloud storage, estimated USD6.54 provisional. Manuscript/source inputs preserved; multi-seed validation deferred.

[Analysis004](../analyses/004-2026-10-04-lr-schedule-intervention/README.md) completed
the first H004 intervention: seed1235,14M/70M, original/held/accelerated decay
from update356. Faster decay reduces raw negativity and brings positive alignment
earlier; held LR has the most negative alignment and lowest validation loss at
both sizes. At70M, positive recovery is temporary. F/D size ordering reverses
under held LR. [The restricted conclusion](../analyses/004-2026-10-04-lr-schedule-intervention/REPORT.md)
concerns schedule-sensitive cancellation, not a learning-quality/stability proxy.
All six artifact sets are verified and workers/volumes removed;88minutes,
estimated USD6.92 provisional (USD15 cap). Three PDFs are complete. Multi-seed
validation is deferred; source evidence and manuscript assets are preserved.

[Analysis003](../analyses/003-2026-10-04-raw-gradient-regimes/README.md) compares
raw-gradient cancellation and loss, with three pilot 1x2 figures and a
[completed replication audit](../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md).
Larger models have lower validation loss, more negative adjacent pairs and higher
unconditional negative intensity, but smaller magnitude conditional on negativity,
in 3/3 seed blocks. Net all-pair raw interaction is positive in 9/9 runs; lag-one
truncation misses 97.90-98.57% of squared cumulative gradient norm. The restricted
[manuscript conclusion](../analyses/003-2026-10-04-raw-gradient-regimes/REPORT.md)
is retained. H003's dimensionality/performance mechanism remains untested;
manuscript and source-run evidence are unchanged.

[Observation002](../analyses/003-2026-10-04-raw-gradient-regimes/observations/002-schedule-and-recovery.md)
extends Analysis003 with the common LR schedule and phase-specific raw alignment.
Decay/recovery timing is compatible with H004, but the 70M trajectory returns to
negative alignment at fixed floor LR. Schedule and training progress remain
confounded in that observational check; the separate intervention is Analysis004.
The [recommended experiment](../analyses/003-2026-10-04-raw-gradient-regimes/SCHEDULE_EXPERIMENT.md)
branches update356 checkpoints at14M/70M into original, held and accelerated
decay. Analysis004 executes the user's selected first seed; the original
three-seed recommendation is reserved for later validation.

Runs 005-010 completed both additional seed blocks. [Analysis002](../analyses/002-2026-10-04-scale-seed-replication/README.md)
compares three seeds per size: raw cosine: 3/3 seed blocks; raw B/A: 3/3 seed blocks; adaptive cosine: 3/3 seed blocks; adaptive B/A: 3/3 seed blocks.
All new artifacts are verified; all six workers/volumes removed; estimated $10.11 provisional.
Initial/order seeds vary together; order is paired within seed. Three seeds support a
replication check, not a predictive scaling law.


Runs 003 (31M) and 004 (70M) completed both full MiniPile passes, 712 updates each,
with the paired raw-gradient/adaptive-direction metrics from Run 002 (14M).
All new artifacts were retrieved and verified; both workers and their volumes
are removed. [Analysis001](../analyses/001-2026-10-03-model-scale-geometry/README.md)
owns the matched scale comparison and four publication figures.

| Model | Raw median cosine | Adaptive median cosine | Raw final B/A | Adaptive final B/A |
| --- | ---: | ---: | ---: | ---: |
| 14M | 0.007220 | 0.966250 | 0.337132 | 0.704780 |
| 31M | −0.086252 | 0.952865 | 0.330751 | 0.665558 |
| 70M | −0.160566 | 0.931795 | 0.309380 | 0.609436 |

Full-pass alignment/ratio ordering is consistent across these three sizes.
Raw A/B decrease, adaptive A/B increase; raw median norms and phase-specific
raw cosines are not uniformly monotonic. H002's predictive scaling claim remains
inconclusive: one seed per size, no pre-specified functional form or held-out size.
The earlier one-seed comparison above is retained for reference; completed
replication is summarized at the top. The adaptive direction excludes LR/decay.
Main plots omit the final partial batch;
full data and terminal statistics retain it. Estimated new-run cost USD3.33
is provisional; no task-owned cloud storage remains.

## Earlier evidence

Run002 supports H001 descriptively: adaptive adjacent alignment and B/A exceed
raw-gradient values, while A and B retain a gap. [Run001](../runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/README.md)
retains the first raw-gradient baseline and its temporal-interaction analysis.
Its replay differs from Run002; use Run 002's paired streams for the scale comparison.
No stability, causal learning benefit, or full SGD identity is established.
SGD remains deferred. Original manuscript assets are preserved.

## Navigation

- [Hypotheses](HYPOTHESES.md), [runs](../runs/INDEX.md),
  [analyses](../analyses/INDEX.md), [findings](findings/README.md)
- [Definitions](DEFINITIONS.md), [methods](METHODS.md), [metrics](METRICS.md),
  [model/data protocol](DATA.md)
- [Workflow](WORKFLOW.md), [compute and ETC](COMPUTE.md), [Runpod](RUNPOD.md),
  [artifact contract](ARTIFACTS.md), [plotting](PLOTTING.md)
- [Manuscript](MANUSCRIPT.md), [review context](REVIEW_CONTEXT.md),
  [bootstrap provenance](BOOTSTRAP.md)
- [Surgical changes proposal](../manuscript/2026-10-07-surgical-changes-proposal/README.md):
  evidence, draft changes, and proposed momentum/LR experiments; planning only.

Next run: 038; analysis: 013; hypothesis: H012. Preserve executed inputs and raw evidence.
