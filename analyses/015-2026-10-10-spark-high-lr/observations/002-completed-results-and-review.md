# Observation002: completed high-LR results and final review

Both authorized Spark runs completed on 2026-10-11: Run053 at LR0.02 and
Run054 at LR0.04. Each recorded all712 accepted updates, 1,491,711,416 input
tokens and 1,490,983,041 supervised targets, with zero overflow retries.
Initial and final validation each include all339 sequences and693,329 targets.
The queue finished at10:52:15 UTC (07:52:15 America/Sao_Paulo); artifact
generation finished at10:52:17 UTC. Training ETC is zero. Full passes took
about5 hours each, with mean full-update times25.229/25.287 seconds.

| Run | Peak LR | Final validation loss | Last100 full-update training loss | Negative raw adjacent pairs | Negative momentum adjacent pairs | Final raw B/A | Final momentum B/A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 023, reference | 0.01 | 6.636489 | 6.309059 | 487/711 (68.50%) | 0/711 | 0.044843 | 0.653121 |
| 053 | 0.02 | 6.634378 | 6.501261 | 455/711 (63.99%) | 0/711 | 0.027739 | 0.342280 |
| 054 | 0.04 | 5.191948 | 5.476455 | 445/711 (62.59%) | 0/711 | 0.022742 | 0.307993 |

LR0.04 finishes with lower validation and late training loss than both other
conditions in this matched comparison, despite its early loss rebound.
LR0.02 finishes close to LR0.01 in validation loss, with higher late training
loss. Momentum full-prefix B/A decreases as peak LR increases across these
three trajectories; adjacent momentum cosine remains positive throughout.
Raw adjacent cosine is often negative in every condition. These observations
do not establish an optimal LR, a stability threshold or a causal role for
alignment. One initialization, seed1234 order and finite pass provide no seed
uncertainty; larger LR also scales the weight-decay displacement.

Sources: [full-precision summary](../summary.csv),
[exact plotted series and source hashes](../comparison_data.json),
[terminal queue](../queue_state.json), and each owning run's full-pass
summary/verification/events and retrieval receipt. The terminal rows are
independently checked against all local raw-loss/raw-gradient/momentum scalar
events. Late loss is supervised-token weighted over updates612-711; partial
update712 is excluded from that average. Negative-pair counts use all711
defined adjacent pairs, including the partial endpoint. Final B/A includes
all712 vectors. No curve smoothing enters this table.

Figures: [compact loss and norms](../figures/loss_and_norms_by_learning_rate.pdf)
and [full training dynamics](../figures/training_dynamics.pdf). Their captions,
stream definitions, smoothing windows and coverage are specified in
[Observation001](001-high-learning-rates.md); the generating scripts remain
[03_plot.py](../03_plot.py) and [04_record_results.py](../04_record_results.py).
Loss is unsmoothed. Norm panels show faint raw values, a trailing10-update
arithmetic mean and fill to unsmoothed stored momentum, on common logarithmic
axes. Full dynamics adds adjacent cosines and cumulative geometry. Fill denotes
a magnitude gap, not uncertainty.

All source hashes, original remote output hashes, scalar reductions and coverage
passed local checks. Both final PDFs were rendered with Poppler and visually
inspected: titles, labels, legends, plotted trajectories, endpoint markers and
footnotes are readable and unclipped. Review render hashes and PDF hashes are
recorded in [local verification](../local_artifact_verification.json).
The remote figure-verification record retains its pre-inspection status;
this local record supplies the completed visual review.

Local retrieval finished at10:55:20 UTC. Each run's60 inventoried files passed
byte-size/SHA-256 checks, totaling15,876,432,449 bytes across both runs.
All recovery states remain on Spark and locally, outside Git. Spark was idle
after completion. No cloud resource or rental billing was introduced.
The [remote result inventory](../result_inventory.json) describes the original
downloaded payload preserved in commitfb6435e; subsequent local README/index
edits record closeout and do not change scientific inputs, data or PDFs.
