> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [README.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md).

# MLP interventions: ongoing analysis (064 / 065)

This is the continuing workspace for the MLP intervention study. Analysis 064
contains its preserved first-stage evidence; 065 continues the same analysis,
not a separate scientific study. Keep subsequent iterations here until the user
requests another analysis. Distinct executed configurations retain distinct run
IDs. Every intervention is compared with the **unchanged Adam baseline**.

| Arm | Complete set of baseline changes | Evidence |
|---|---|---|
| Baseline | None; Adam LR 0.001, all layers updated | Qualified retained control in 064 |
| I1 | Freeze first-linear weights and biases from state 448 | Run 244 |
| I2 | I1 plus Adam LR 0.001 -> 0.002 from state 500 | Run 245 |
| I3 | I1 plus Adam LR 0.001 -> 0.02 from state 500 (10x I2) | Run 246 |
| I4 | I1 plus Adam LR 0.001 -> 0.01 from state 500 (half I3) | Run 247 |
| I5 | I1 plus Adam LR 0.005 from state 500 | Run 248 |
| I6 | I1 plus Adam LR 0.0025 from state 500 | Run 249 |
| I7 | I1 plus Adam LR 0.00125 from state 500 | Run 250 |
| I8 | I1 plus Adam LR 0.000625 from state 500 | Run 251 |
| I9 | First-layer LR 0.0005 from 448; later layers 0.001; no freeze | Run 252 |
| I10 | First-layer LR 0.00025 from 448; later layers 0.001; no freeze | Run 253 |
| I11 | All-layer LR 0.0005 from 448; no freeze | Run 254 |
| I12 | All-layer LR 0.00025 from 448; no freeze | Run 255 |
| I13 | First-layer LR 0.001; later layers 0.0005 from 448; no freeze | Run 256 |
| I14 | First-layer LR 0.001; later layers 0.00025 from 448; no freeze | Run 257 |
| I15 | Discard large first-layer Adam coordinate increments from 448; prefix q90 threshold | Run 259 |
| I16 | Cap large first-layer Adam coordinate increments from 448; prefix q90 threshold | Run 260 |
| I17 | Cap each first-layer neuron's preactivation displacement RMS; prefix q90 budget | Run 261 |
| I18 | Same neuron displacement rule, prefix q75 budget | Run 262 |
| I19 | Same neuron displacement rule, prefix q50 budget | Run 263 |
| I20-I24 | Uniformly scale first-layer proposals to match I15-I19's respective applied norm schedules | Runs 264-268 |

For I2-I8 the first changed-rate update is 500 -> 501. No optimizer reset or unfreezing.
All 5000 CIFAR-10 training examples, ordering, initialization, objective and
remaining optimizer/precision settings match. I2 through I8 branch from the same
state 500; none waits until the end of another intervention. Comparisons between interventions
describe incremental LR effects and supplement the baseline comparisons.

I9-I14 instead branch from the **untreated full state 448**, with all layers
trainable and first changed update 448 -> 449. They test selective damping of
first-layer updates against global and later-layer-only LR controls. Their
complete interventions contain no freezing and no LR change at 500.

I15-I24 also start from untreated state 448. All native Adam LRs remain 0.001;
raw gradients, moments and clocks advance normally. The intervention transforms
only the proposed first-layer weight/bias increments after native Adam. Run 258
calibrates fixed thresholds from untreated updates 416-447, before the fork.
The neuron criterion uses all 5,000 samples. I20-I24 are paired experimental
controls; each matches its selective arm's actual first-layer norm at the same
update, within measured FP32 rounding, and may amplify its own native proposal.

## Results and artifacts

I1 and I2 suppress the baseline's large loss excursions and output
motion. Across 448-800, however, mean training loss is **22.27% higher for I1**
and **20.58% higher for I2** than the unchanged baseline. Endpoint loss at 800 is
10.32% / 13.04% lower, illustrating why the endpoint alone is insufficient.
I2 produces more small output reversals than I1 without restoring large spikes.

**I3 at LR 0.02 causes severe overshoot:** loss rises from 0.028335 at 500 to
107.099 at 504 and recovers only to 0.308132 at 800 (10.19x baseline). First-layer
parameters and representations remain exactly fixed. Later-layer updates can
therefore cause large instability despite that freeze at sufficiently high LR.
The run stays finite; this is degraded training, not numerical overflow.

**Halving stopped at 0.000625 (I8):** all 300 loss transitions over 500-800
decrease. At 0.00125, visible spikes are already gone; only a 0.0060% first-step
bump remains. At 0.0025, a delayed peak reaches 0.0724478 at 707. The lower rates
must be judged over the full window, not only immediately after the change.
The recorded strict rule permits no upward step above 1e-8 and no loss above
loss500 + 1e-8. This is a finite-window criterion, not an optimal-LR estimate.

**Selective LR reductions retain feature learning:** I10 (first layer at 0.25x,
later layers unchanged) lowers mean loss 448-800 by 7.06% and positive loss
variation by 99.62%. First-hidden reversals fall 350/353 -> 1/353. Its last-32
mean remains 15.79% above baseline, so it does not win on every learning metric.
First-layer 0.5x develops a larger delayed peak at 673. Global 0.5x has the best
full-window mean (-17.05%) but a rising excursion at 800. Later-layer reductions
improve average loss while retaining most reversals. See observation005 for the
matched controls, sustained-target results and limits.

- [Four-page baseline comparison](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/01-frozen-first-linear-double-lr-cifar10-mlp.pdf):
  I1-I4 on pages 1/2 and I5-I8 on pages 3/4, each overlaid with unchanged baseline.
  Overview pages: eight diagnostics over 448-800, including eta kappa hat,
  gradient norms and LR. Companion pages: five representations over 500-550.
  Scales match across columns and page groups.
- [Observation, definitions and caveats](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/001-frozen-lr-double.md)
- [I3: the 10x LR extension and severe overshoot](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/002-frozen-lr-0p02.md)
- [I4-I8: LR halving and the strict stopping rule](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/003-lr-halving.md)
- [Separate I6/I5 comparison](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/02-i6-vs-i5-cifar10-mlp.pdf): I6 left,
  I5 right, each against unchanged Adam over 448-800. Independent linear loss
  axes; shared scales for the other seven diagnostic rows.
  [Observation and reproduction](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/004-i6-i5-comparison.md).
- [New selective-LR comparison](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/03-selective-layer-lr-cifar10-mlp.pdf):
  three pages, two columns per page (0.5x / 0.25x), comparing first-layer,
  global, and later-layer-only reductions against unchanged Adam. Ten rows
  include first-hidden motion; loss axes are independent.
  [Results, methods and caveats](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/005-selective-layer-lr.md).
- [Verified halving summary](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/halving_summary.json), [recorded rule](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/halving_plan.json)
- [Numerical results and provenance](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/comparison.json), [window summaries](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/metrics.csv)
- [Run 245 verification](../../../../../../runs/245-2026-11-07-mlp-frozen-lr-double/verification.json)
- [Run 246 verification](../../../../../../runs/246-2026-11-07-mlp-frozen-lr-0p02/verification.json)
- [First-stage analysis and SVD evidence](../../../../../../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md)

Reproduce with `python analyses/065-2026-11-07-mlp-frozen-lr-double/01_reduce.py`
then `02_figures.py` in the same directory, using the verified retained artifacts.
Native continuation qualification, 66 source identities, 22 retrieved files,
independent dense reductions and baseline duplication checks pass. Both PDF pages
were rendered and reviewed. Run 245 took 7.76 s and Run 246 took 7.96 s on owned
Spark; both workers exited. Their retained directories use 2.021 / 1.911 GB,
respectively, with verified local copies and zero cloud cost. Run 246 adds 67
verified source identities and 22 retrieved files. Plot scales are explicitly
logarithmic or symmetric-log where needed to show I3's much larger excursion.

Runs 247-251 add 1,505 intervention updates in 40.62 s total, 67 verified source
identities per run and 110 verified downloaded files. Their full fork states are
exactly identical to I2's fork. All workers exited; 10.034 GB retained for these
five runs on owned Spark with verified local copies, no cloud cost. Stopping was
confirmed from local raw losses; all four final PDF pages were reviewed.

Runs252-257 complete six prescribed LR-location branches, 353 updates each,
with exact untreated448-512 qualification and full fork/state matches. All
67 source identities and 22 retrieved files per successful run, independent
dense reductions, 50832 plotted values and three new PDF pages pass review.
Workers exited after49.75s total successful execution; peak allocation400.8MB.
13.720GB retained on owned Spark, including one rejected qualification attempt,
with verified local copies and no cloud charge. Regenerate the new results with
`07_reduce_layer_lr.py` and the new PDF with `08_figures_layer_lr.py`.

## Selective update gates: I15-I24

All three proposed methods are complete, including three budgets for the neuron
criterion and five uniform controls matched to the selective first-layer norm
schedules. I18 (neuron q75) improves mean loss 448-800 by 10.42% and positive loss
variation by 79.37%, while its tail mean remains 5.68% above unchanged Adam.
It modifies 15.37% of neurons per update on average and beats its paired norm
control by 4.63% in mean loss. Hidden/output changes still reverse on all 353
updates: this damps amplitude rather than removing reversals. Coordinate discard
worsens mean loss; coordinate capping gives a small benefit. Tighter neuron
capping is not monotonic, and I24 (uniform q50 norm schedule) has the lowest new
full-window mean. One seed/fork; no general superiority or generalization claim.

- [Methods and baseline comparisons](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/04-selective-update-methods-cifar10-mlp.pdf): three columns, ten diagnostic rows.
- [Neuron budgets and matched-norm controls](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/05-selective-update-norm-controls-cifar10-mlp.pdf): two pages, baseline in every column.
- [Results, definitions, captions and caveats](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/006-selective-update-gates.md).
- [Verified numerical record](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/selective_comparison.json), [window summaries](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/selective_metrics.csv), [compute closeout](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/selective_compute_closeout.json).

Regenerate with `11_reduce_selective.py` then `12_figures_selective.py`.
All source identities, 270 retrieved files, independent gate/dense checks,
74,130 plotted values and three rendered pages pass; Figures 01-03 preserved.
Workers exited after 385.42 s total including calibration;
21.907 GB retained on owned Spark with verified local copies, zero cloud cost.
Keep further interventions in this analysis; the H048 extension follows below.

## Representation-informed transfer: H048

Runs 269-285 extend this same analysis to ResNet/CIFAR-10 and Transformer/WikiText-2,
with additional MLP policy dissection and three-initialization replication.
All comparisons retain unchanged Adam controls and the complete native training
objectives. The user authorized the runs and a USD 20 total Runpod ceiling.

**Complete:** distributed local affine-displacement caps are the strongest
transferable optimization candidate in this screen. The common prefix-q50
rule lowers mean training loss by 22.85% (MLP seed 0, 448-1152), 16.22%
(ResNet, 264-576), and 26.00% (Transformer, 552-928). MLP seeds 1/2 improve
19.43%/32.08% without retuning. All gradients, Adam moments and clocks advance;
all layers remain trainable. Every effect uses its own unchanged Adam control.

The cap reduces loss excursions while reversals persist or become more frequent.
Negative alignment alone is therefore not a sufficient intervention target.
Terminal test loss is worse under this cap in every setup; this is an
optimization result, not a generalization improvement. Diagnostic-triggered
gates and past-axis removal can fail severely. Doubling the Transformer embedding
LR is nearly neutral in mean loss; quadrupling it worsens the mean. Adding 4x
to the cap worsens mean loss 22.47% relative to the same-worker cap-only arm.

- [Design and amendments](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/007-cross-architecture-plan.md).
- [Methods, MLP replication, transfer results and figure captions](65-008-cross-architecture-interventions.md).
- [What the MLP negative controls and exact loss accounting show](65-009-what-the-negative-controls-show.md).
- [The geometry and limits of the local displacement cap](65-010-local-displacement-cap-geometry.md).
- [Completed Spark process/log verification](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_spark_closeout.json).
- [Selective entry criteria across architectures](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/06-selective-transfer-diagnostics.pdf): three pages.
- [Distributed caps, architecture traces, replication and controls](../figures/07-distributed-displacement-and-controls.pdf): six pages.
- [Adaptive rules, negative controls and embedding rates](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/08-adaptive-and-embedding-tests.pdf): five pages.
- [Aggregate results](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_summary.json), [compute/storage](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_compute_closeout.json), [cloud cleanup and provisional billing](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_cloud_closeout.json), and [visual review](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/transfer_visual_review.json).

All source/retrieval identities, 210 retained arm-state audits, original-hardware
representation/probe replay and 14 rendered pages pass. There are 68 new
continuations (35,452 updates) and two authenticated reused controls across 18
scientific workers. All five task pods are deleted and no task cloud storage
remains. The conservative compute/storage estimate is USD 18.75, below the
USD 20 ceiling; provider billing is provisional. All Spark task processes exited.
Raw evidence remains locally (41.316 GB unique files) and on owned Spark;
[artifact storage](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/artifact_storage.json) records local NTFS junctions/hardlinks.
Reproduction commands and measurement limits are in observation 008.

New figures use numbers 06-08; earlier figures remain byte-identical. Scientific
inputs are frozen. Keep further work in this analysis; no new analysis ID was
allocated. The next unused run is 286 and the next unused hypothesis is H049.

## Local displacement cap technical report

The [eight-page report](../report/local-displacement-cap-report.pdf) consolidates
motivation from representation-instability tracing, the local cap algorithm,
entry/distributed scopes, probes and training setup, cap-only results and MLP
replication. Its [LaTeX source](../report/local-displacement-cap-report.tex) includes
the numbered algorithm. Terminal test-loss deterioration and the distinction
between local and total representation movement remain explicit.
Notation follows the manuscript: raw gradient `g_t`, first/second moments
`m_t`/`v_t`, Adam direction `d_t`, and raw-gradient cosine `rho_t`. Proposed
and capped parameter increments are named separately. The report and figure
labels explicitly map completed state `s` to outgoing update `t=s+1`.

**Revision 001** applies [Feedback 001](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/report/feedback/001/README.md): fixed
inputs and flattening, precise tracing findings, explicit matrix/score/cap
dimensions, calibration by update step, per-unit scaling and forward-pass cost.
The original annotated PDF and all 11 annotations are preserved in the numbered
archive, with dispositions for ten complete comments and one unfinished note.
The next feedback round is 002; the report filename stays stable for review.

New manuscript-style figures show [entry caps](../figures/09-local-cap-entry-manuscript.pdf),
[distributed caps](../figures/10-local-cap-distributed-manuscript.pdf), and
[MLP replication](../figures/11-local-cap-mlp-replication.pdf). Loss uses logarithmic
axes with independent scales per column; eta-kappa-hat uses linear axes.
Only local cap variants and unchanged Adam appear. Existing figures are preserved.

[Observation 011](65-011-local-cap-technical-report.md) records sources,
captions, coverage, caveats and reproduction. All 61,450 plotted values, 12 table
rows, eight report pages and three standalone figure pages pass checks and visual
review. No new training, cloud work or scientific allocation.

## Distributed cap quantile and LR ablation (H049)

**Completed:** Runs 286-288 contain 54 prespecified continuations (three initializations,
five quantiles and three LRs, plus same-LR uncapped controls). Fixed 1x-calibrated
budgets make 2x/4x LR much less disruptive, but no quantile improves its across-seed
mean training loss by increasing LR. q25 tolerates high LR best; q50 at 1x has the
lowest observed mean. Tight caps retain smaller oscillations and can worsen test
loss. This supports controlled large excursions, not an acceleration or
uniform generalization claim.

- [Results, method, captions and limits](65-013-distributed-cap-quantile-lr.md).
- [Training/test curves](../figures/12-distributed-cap-quantile-lr-mlp.pdf) and
  [three seed-resolved diagnostic pages](../figures/13-distributed-cap-lr-seed-diagnostics.pdf).
- [Readable statistics](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/cap_lr_statistics.md), [aggregate CSV](../data/cap_lr_aggregate.csv),
  [per-seed CSV](../data/cap_lr_per_seed.csv), and [LaTeX table](../tables/cap_lr_table.tex).

All 38,070 updates, 162 retained audits, six exact historical reproductions,
310,660 plotted values and four rendered pages pass. Training took 34.34 min on
owned Spark; no cloud cost. Task processes exited. After a local disk-limit
recovery, complete raw evidence stays on Spark (7.978 GB), while verified compact
outputs, scalar histories and audit records are local. Each run records its full
remote inventory. [Transfer](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/cap_lr_compact_retrieval.json), [storage adjustment](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/cap_lr_local_storage_adjustment.json),
and [compute closeout](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/cap_lr_compute_closeout.json) preserve the audit trail.

`40_plot_cap_lr.py` regenerates figures locally from the verified reduced data.
`42_export_cap_lr.py` performs a full reduction on the retained Spark workspace
and retrieves compact results. Previous figures/report and scientific inputs are
unchanged. Continue this same analysis; next unused run 289 and hypothesis H050.

[Cap-quantile ablation](../figures/14-distributed-cap-quantile-ablation-mlp.pdf)
shows connected means and sample-SD error bars in three LR columns with a shared
logarithmic loss axis. [Observation014](65-014-cap-quantile-ablation-figure.md)
records sources, reduction, caption and checks. Existing data only; no new training.
