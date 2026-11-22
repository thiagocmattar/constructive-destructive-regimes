# Evidence freshness and consistency audit

- **Current authority:** [PLAN](../PLAN.md), 14 November 2026. H051 closes the agreed matched-intervention and reporting gaps; manuscript integration remains separate.
- **Preservation:** original manuscript/review/author inputs remain unchanged. Canonical historical evidence is retained; six superseded V-C selections are archived byte-identically in [background](05-adam/c-intervention/background/README.md).
- **Current displays:** Figures 1–6 and the temporal companion retain the verified completed-step convention and selected ResNet 390–450 window. V-C now uses Figures 7–9: four-family loss matrix, spike/learning tradeoff and matched q50 heatmaps. Seven intervention PDFs / 20 pages are visually inspected.

## Scientific verification

- **H051 coverage:** 36 conditions, 11,904 updates, 11,940 full-gradient boundaries. 108 exact raw-gradient/probe/native-update audits; 27 cap-bound audits. Every saved terminal state additionally reproduces its full training gradient/loss and all complete terminal held-out splits: 36 gradient and 48 split checks.
- **Matched state:** full model/optimizer/RNG/data fork, exact predecessor replay and common eight-update fingerprints within model. The new ResNet control also reproduces 187 historical losses and 732 selected-window stage scalars; this does not establish full historical tensor identity for every model.
- **Calibration:** 8 untreated outgoing states f−8,…,f−1; frozen module budgets; first treated update f→f+1. Complete score pools, quantile calculations, zero-budget convention, actual stored-increment bounds and activity are retained.
- **Reductions:** mean excludes the common fork; PV sums all positive post-fork increments; final and last32 remain distinct. Excursions keep their original pre-rise reference and first return, with carried/right-censored events explicit. Independent fixture and raw-stream reductions pass.
- **Motion:** every fixed-probe coordinate contributes to RMS/energy; finite consecutive-change cosines define reversal fractions. MLP's output-space squared-loss identity is independently checked. Cross-entropy is not assigned that identity.
- **Training coverage:** MLP 5,000 images; ResNet 50,000 images in 50 intact 1,000-image normalization groups; Transformer 2,088,600 supervised tokens in 59,680 sequences of up to 35 positions.
- **Held-out coverage:** CIFAR-10 full 10,000-image test; WikiText-2 full 217,630 validation and 245,550 test tokens. ResNet's 10 test groups and Transformer's 7,020 test sequences are objective blocks, not loss-reduction denominators. Five records/arm include a shared fork: 147 distinct evaluation states / 196 full split evaluations.
- **Cost interpretation:** synchronized gradient/probe-capture/Adam/controller/evaluation timers; other wall time includes cosine/RMS reductions, scalar diagnostics and I/O. Memory is process high-water allocation through an arm. Different cloud hosts preclude a method-speed ranking.

## Retention and publication checks

- **Retention:** 33.842 GB of raw evidence on owned Spark; sizes and SHA-256 verified before every task Pod deletion. No retained cloud volume. Provisional compute plus temporary storage **$26.07**, below $50. [Closeout](05-adam/c-intervention/data/matched-compute_closeout.json).
- **Publication sources:** 151 registered assets from 150 unique source files. Numerical plot coordinates, PDF hashes/page bounds/embedded fonts, all 20 intervention render pages and editable TeX are checked. [Figure audit](05-adam/c-intervention/data/matched-visual_verification.json); [package audit](verification.json); [snippet checks](snippet_verification.json).
- **Historical scope:** earlier H050 conventional results and the three-initialization MLP cap/LR study retain their own forks, runtimes and endpoints. They are not reused as cells of the new matrix.
- **Limits:** one initialization/model and one fixed window; unequal treatment strengths/tuning grids; no held-out selection, seed confidence intervals, universal superiority, stochastic deployment or generalization benefit. Package checks do not re-prove every historical computation or literature theorem.

## Remaining work

- Integrate the prepared evidence/captions into the manuscript, then compile and review the full paper.
- Keep stronger forecasting, causal-source, spike-specific-mode and method-superiority claims conditional on their separate tests. Extra initializations and cap x LR remain [deferred](05-adam/c-intervention/TODO_ABLATIONS.md).
- Resolve the current TNNLS route/format before any submission, as recorded in PLAN.
