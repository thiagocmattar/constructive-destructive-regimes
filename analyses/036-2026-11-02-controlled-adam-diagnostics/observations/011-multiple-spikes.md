# Multiple-spike windows with linear relative norms

Date: 2026-11-05. Requested extension of the retained single-spike figure for
reconsidering the Adam section. The original figure, manuscript copy, TeX and
raw evidence are unchanged.

Output: [multiple-spike figure](../figures/adam_manuscript_spikes_multiple.pdf).
Generator: [13_plot_multiple_spikes.py](../13_plot_multiple_spikes.py).
Provenance and exact plotted arrays: [manifest](../multiple_spikes_manifest.json).
Reproduce from the repository root:

```powershell
python analyses/036-2026-11-02-controlled-adam-diagnostics/13_plot_multiple_spikes.py
```

## Sources and selection

Only existing Run215 histories are used, at the same highest tested rates as
the original figure: MLP 0.001; ResNet-32 and Transformer 0.003. Each model
has one seed-0 initialization and 1,024 full-objective Adam updates, without
clipping, weight decay or scheduling. Objectives are mean half squared error
on the prepared 5,000 CIFAR-10 images, mean cross entropy on all 50,000
CIFAR-10 training images, and token-weighted NLL on all 2,088,600 prepared
WikiText-2 targets, respectively. Parameter scope and numerical precision
are inherited from Run215; there is no new gradient computation or training.

The author requested approximate MLP 450-750, ResNet 200-450 and Transformer
500-850 windows, with several spikes, pre-rise context and final recovery.
Endpoints were selected manually from the loss traces and the existing
fixed-reference event boundaries; no cosine, norm or projected-change filter
was used. In particular, ResNet's largest spike is still recovering at 450,
and Transformer's at 850.

| Model | Displayed states, inclusive | Displayed observations | Complete >10% events | Last large event: pre-rise / peak / first recovery |
| --- | --- | ---: | ---: | --- |
| MLP | 465-744 | 280 | 9 | 689 / 703 / 729 |
| ResNet-32 | 215-495 | 281 | 6 | 415 / 422 / 482 |
| Transformer | 525-865 | 341 | 4 | 787 / 794 / 855 |

All endpoints lie outside open fixed-reference excursions, including small
fluctuations. MLP ends at 744 to include recovery from its last small rebound
before the next large event begins at 745. Each panel begins after the
preceding event's recovery and before a new rise. The windows retain the
original largest recovered spikes and add surrounding events. Counts refer
to complete fixed-reference excursions with peak increase strictly above
10%; multiple local maxima within one excursion remain one event.

## Reduction and caption

All 18 curves / 5,412 plotted values are unsmoothed, with every recorded state
shown and no temporal averaging or downsampling. The four rows retain loss,
raw-gradient projected change, consecutive raw-gradient/first-moment cosine,
and the two relative norms. Each norm is divided by its own value at the
first displayed state, as in the original figure. **Both norm curves now use
a linear y-axis starting at zero.** Loss retains its logarithmic y-axis;
projected change and cosine remain linear. Relative norm heights across
models have different normalization denominators and are not absolute-norm
comparisons.

Suggested caption: Repeated loss excursions during full-batch Adam at the
highest tested learning rate for each model. Rows show training loss,
normalized projected raw-gradient change, consecutive cosines of raw gradients
and stored first moments, and their relative norms. Blue solid and orange
dashed curves denote gradients and first moments in the bottom two rows.
Loss uses logarithmic axes; relative norms use linear axes and each stream
is normalized by its own first displayed value. Gray bands span pre-rise
state to loss peak for every complete >10% excursion in the window. All
observations are unsmoothed. At parameter state s, loss is evaluated at
theta_s and vector diagnostics use g_(s+1), m_(s+1); the horizontal coordinate
counts completed optimizer updates. Dashed horizontal references mark
projected change two and cosine zero. One initialization per model.

Unlike the original micro-plots, the gray bands begin at the formal fixed
pre-rise state, rather than at the +10% crossing. They include all 19
qualifying excursions in these windows, not just the largest event. There
is no single horizontal reference loss shared across the different events.

## Coverage, uncertainty and interpretation

These are retrospectively selected illustrations of repeated temporal
association. There are no independent-seed uncertainty intervals, predictive
scores, non-spike controls, causal interventions or new Hessian measurements.
The projected-change quantity retains its recorded raw-gradient meaning
along Adam's adaptive path; the horizontal reference at two is not an
established Adam stability boundary. The figure alone does not validate
second-order curvature, smoothness estimates or prospective spike prediction.

## Verification

The generator verifies source SHA-256 hashes against the event dataset's
manifest, all 18 curves against the archived unsmoothed arrays, loss equality
between boundary and gradient logs, complete source clocks, the projected
change identity, and all 19 highlighted events' peak/recovery boundaries.
It also checks normalization, linear norm axes, exact plotted artist arrays,
embedded fonts, vector-only output, text containment and legend clearance.
The 7.16 by 5.6 inch PDF was rendered with Poppler at 2,200-pixel width and
visually reviewed: all twelve panels, ticks, labels and the shared legend
are legible, with no overlap or clipping. Scratch renders remain ignored.

The legacy `03_verify_loss_spikes.py` was also attempted. Its 41 detector
checks pass, but its full audit stops at an already-stale README output hash
in `spike_dataset/manifest.json`. The current task does not rewrite that
historical manifest or claim this legacy audit passed. The new figure's
source-data hashes, source-array checks and event-boundary checks pass
independently of that documentation hash.
An independent read-only audit also matches all 812 event boundaries across
the nine trajectories using the legacy streaming detector, verifies every
scientific input/output hash in the event manifest, and checks all 5,412
new plotted values directly against the raw logs. Only the README hash
differs; neither the event data nor its manifest was modified.

No new run, analysis ID, hypothesis, cloud resource or billable execution.
