# Artifact and source map

Paths below point to existing evidence; storyboard IDs are proposed in
[PLAN.md](../PLAN.md). Replotting belongs in the source analysis, with a new
observation and independent numerical/PDF verification. Do not overwrite the
author's supplied inputs or historical publication figures.

## 1. Main displays and where to find them

| Plan element | Existing source and intended placement | Reduction / generation / limits |
|---|---|---|
| F1 / E08-E09, GD geometry and progress | Active [loss/proxy/spectral figure](../../draft/figures/analysis024/gd_loss_projected_change_spectral_clean.pdf) and [reversal/displacement figure](../../draft/figures/analysis024/gd_reversal_norm_retention.pdf); Section IV. | [Analysis024](../../../analyses/024-2026-10-17-eos-gd-lr-calibration/README.md), especially observation007 approximation audit; [Analysis056](../../../analyses/056-2026-11-05-gd-paired-displacement/README.md) for all-lag evidence. Choose a compact composite; retain highest-rate approximation failures. |
| F2 / E10, batch effects | Active [three-model batch figure](../../draft/figures/analysis037/sgd_batch_size_compact.pdf); Section IV. | [Analysis037](../../../analyses/037-2026-11-02-sgd-batch-extension/README.md) and [Analysis027](../../../analyses/027-2026-10-30-minibatch-noise/README.md). Distinguish frozen measurements, native sampled gradients, and evolving training paths. |
| Optional lag inset / appendix | Active [backward profile](../../draft/figures/analysis035/gd_backward_compact.pdf). | [Analysis035](../../../analyses/035-2026-11-02-eos-backward-anchors/README.md). Fixed endpoint/varying lag is not a moving adjacent-time plot. |
| F3 / E11, high-LR Adam | [058 Figure00](../../../analyses/058-2026-11-06-intervention-consolidation/figures/00-high-lr-overview-cifar10-mlp-resnet-wikitext2-transformer.pdf); opening of Section V. | [Observation000](../../../analyses/058-2026-11-06-intervention-consolidation/observations/000-high-lr-overview.md), [01_figures.py](../../../analyses/058-2026-11-06-intervention-consolidation/01_figures.py), [02_verify.py](../../../analyses/058-2026-11-06-intervention-consolidation/02_verify.py). One Run215 high-rate trajectory/model; selected retrospectively, full-objective training. |
| T1 / E07 and T3 / E12, timing and event table | **Proposed displays**, based on [055 direction context](../../../analyses/055-2026-11-05-adam-directional-amplification/observations/002-direction-context.md) and [057 complete event analysis](../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md). | P02/P04. Current peak/endpoint statistics cannot silently become pre-update predictors. Preserve all-step/held-out failures in supporting results. |
| A1 / E13, tracing algorithm | [063 METHOD.md](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md), [technical report source](../../../analyses/063-2026-11-07-architecture-instability-tracing/report/instability-tracing-report.tex), [workflow](../../../research/INSTABILITY_TRACING.md); Section V. | Shapes, probes, observer invariance, every-update capture, Gram/SVD computation, timing, costs, null cosine. Window-fitted basis is retrospective. |
| F4 / E14-E15, architectural trace | [ResNet trace](../../../analyses/063-2026-11-07-architecture-instability-tracing/figures/01-resnet-instability-trace.pdf), [Transformer trace](../../../analyses/063-2026-11-07-architecture-instability-tracing/figures/02-transformer-instability-trace.pdf), [MLP450-500 trace](../../../analyses/063-2026-11-07-architecture-instability-tracing/figures/04-mlp-instability-trace-450-500.pdf). | [02_figures.py](../../../analyses/063-2026-11-07-architecture-instability-tracing/02_figures.py), [07_mlp_event_window.py](../../../analyses/063-2026-11-07-architecture-instability-tracing/07_mlp_event_window.py), [08_report_assets.py](../../../analyses/063-2026-11-07-architecture-instability-tracing/08_report_assets.py). Recompose rather than crop a differently fitted window. |
| A2 / E16, cap algorithm | [065 geometry derivation](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/010-local-displacement-cap-geometry.md), [report source](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/report/local-displacement-cap-report.tex); Section VI. | Direct current-input affine movement; post-Adam row/bias scaling; prefix calibration; no SVD in controller. |
| F5 / E18, cap loss and motion | [065 Figure10](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/10-local-cap-distributed-manuscript.pdf), with [Figure07 page2](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/07-distributed-displacement-and-controls.pdf) as motion-layout source. | [22_figures_transfer.py](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/22_figures_transfer.py), [34_local_cap_report_assets.py](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/34_local_cap_report_assets.py). Baseline/cap heatmaps share baseline stage maxima; these maps are not SVD or causal transmission evidence. |
| F6/T4 / E19-E20, controls and replication | [065 MLP replication](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/11-local-cap-mlp-replication.pdf), [quantile/LR summary](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/12-distributed-cap-quantile-lr-mlp.pdf), [seed diagnostics](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/13-distributed-cap-lr-seed-diagnostics.pdf), [quantile ablation](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/figures/14-distributed-cap-quantile-ablation-mlp.pdf). | [Observation013](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md), [39_reduce_cap_lr.py](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/39_reduce_cap_lr.py), [40_plot_cap_lr.py](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/40_plot_cap_lr.py), [43_plot_cap_ablation.py](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/43_plot_cap_ablation.py). Preserve all arms; do not pick q50 using held-out outcomes. |

## 2. Comparator inventory: existing studies are not one matched benchmark

| Study | Fork / conditions / coverage | Main use |
|---|---|---|
| [058 persistent half LR](../../../analyses/058-2026-11-06-intervention-consolidation/figures/01-persistent-half-lr-cifar10-mlp-resnet-wikitext2-transformer.pdf) | MLP512-800; ResNet256-576; Transformer544-928; one initialization/model. | Conditional schedule tradeoffs. Different forks from the cap study. |
| [058 clipping](../../../analyses/058-2026-11-06-intervention-consolidation/figures/05-gradient-clipping-cifar10-mlp-resnet.pdf) | MLP/ResNet, clipping before Adam's moments; no corresponding Transformer full-batch arm. | Input-gradient control; not equivalent to scaling the actual update. |
| [059 beta2 reduction](../../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md) | MLP state512-800, LR.001, beta1.9; beta2 .95/.9/.5; native bias correction with retained moments/clocks. | The missing beta study from the raw note; bounded transition test. |
| [060 Adam to GD](../../../analyses/060-2026-11-07-mlp-adam-to-gd/README.md) | Same MLP state512, several GD rates; optimizer changed. | Optional supporting control; not necessary to add another main narrative. |
| [064 freeze](../../../analyses/064-2026-11-07-mlp-freeze-first-linear/README.md) | MLP from448, first affine weights and biases frozen. | Causal contribution of this composite intervention, with learning tradeoff. |
| [065 layer/selective controls](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/README.md) | MLP from448; selective/global/later-layer LR, coordinate/neuron caps, matched norm controls, directional tests. | Already completed; choose the discriminating controls rather than repeating the exploratory search. |
| [065 distributed transfer](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/008-cross-architecture-interventions.md) | MLP448-1152, ResNet264-576, Transformer552-928; q50, eight untreated calibration updates; same-worker unchanged baselines. | Across-architecture case study; MLP has seed replication, larger models do not. Transformer older half-step and distributed arms also differ in hardware. |
| [065 quantile/LR factorial](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/013-distributed-cap-quantile-lr.md) | Runs286-288; seeds0/1/2; 5 quantiles x 3 LRs plus3 uncapped controls per seed; common fork448, loss448-1152;23 complete test evaluations. | Sensitivity and saturation, not acceleration or an independently selected optimal quantile. |

A new unified table must include objective, training/probe split, normalization
context, hardware/software, optimizer state, calibration, intervention placement,
horizon and held-out coverage. Similar looking plots do not make their studies
matched. Each intervention must retain its own unchanged baseline.

## 3. Contracts for the planned tracing figure

| Case | Full training objective | Fixed representation probe | Fitted outgoing rows |
|---|---|---|---|
| MLP/CIFAR-10 | 5,000 images, half squared error | All 5,000 images | 450-500,51 rows |
| ResNet/CIFAR-10 | 50,000 images, cross-entropy | First intact1,000-image BatchNorm group | 280-310,31 rows |
| Transformer/WikiText-2 | 2,088,600 supervised tokens, cross-entropy | First64 sequences,2,240 supervised positions | 552-600,49 rows |

Keep all sample/token/spatial coordinates. MLP uses linear outputs; the others
use log-probabilities, with correct-label readouts as loss checks. The shorter
MLP fit differs from the earlier448-800 fit. Top-k energy is uncentered
displacement energy, not the fraction of semantic information or loss explained.
The output and internal representations have different coordinate units.

Visual review in this consolidation pass covered 058 Figure00 page1, 063 report
page2 (definitions/algorithm), and 065 Figure07 page2 (heatmaps). Scratch renders
are ignored under `tmp/pdfs/consolidation-review/`. No figure was regenerated or
approved as final publication layout. Future F4/F5 assembly needs numerical
read-back and full-size PDF review after captions/legends are finalized.
