# Complete manuscript rewrite — 14 November 2026

- **Request:** write the complete paper from the agreed consolidation plan, at publication quality, while keeping all work inside consolidation for later author review.
- **Deliverables:** [main manuscript](../../draft/manuscript.pdf), 14 IEEE-style pages; [supplement](../../draft/archive/2026-11-18/supplementary.pdf), 33 pages; editable TeX, bibliography, generated tables, and reproducible build/verification scripts.
- **Scientific direction:** unchanged title, Sections I–VI, III's theoretical spine, IV's four figures, and V characterization → tracing → intervention. The four contribution cards in [PLAN](../../PLAN.md) govern the introduction.
- **Inputs:** all four human notes, the supplied [review skill](<../../human/review-skill/SKILL (2).md>), the current manuscript and research methods, section-indexed evidence, and the annotated local-displacement report. The latter informs explicit matrix dimensions, fixed-probe coordinates, one budget per module, and the practical cap computation.
- **Scope:** manuscript composition and verification only. No training, cloud resources, new empirical claims from synthetic checks, push, or submission. The prior V-C excerpt remains unchanged.

## Claim-to-source audit

| Manuscript claim / placement | Retained source and reduction | Interpretation retained |
|---|---|---|
| III: gradient interaction, path-average curvature, noise, finite-step loss | Existing definitions and equations; explicit smooth-path integration and trapezoidal residual derivations in S1 | BB1 equivalence credited; directional scalar distinct from largest eigenvalue; smoothness and sampling assumptions explicit; incoming response distinct from outgoing prediction. |
| IV: nine GD cases; positive all-pair accumulation with negative adjacent interaction in seven | `evidence/04-gd-sgd/data/24-section_iv_review_checks.json` and `56-endpoints.csv`; generated Table S1 | Exact trajectory reductions, not independent replications. Backward-lag continuations are separate qualified trajectories. |
| IV: 546 sampled continuations; 1,188 HVP pairs | Analyses 027/037 observations and `analysis027_hvp_errors.tex`; fixed-state and moving-state studies separated | Initialization blocks, repetitions, sampled-unit definitions, runtime confounding, and offline full-gradient noise correction disclosed. |
| V-A: severity, false alarms, lead time | `severity-10pct.csv`, `next-step-signs.csv`, and six-page `adam_forward_and_severity.pdf` | Precision denominators include every flag. Individual-step, net-change, within-window, and past-memory targets remain separate. No prospective predictor claim. |
| V-B: concentrated functional movement and shared temporal structure | Current 451–501 / 390–450 / 553–601 fits; revised ResNet `resnet_390_450_results.json`; `spectral_controls.json`; observations 63-007/008 | Current larger ResNet excursion in both figures. Exact coordinates, incoming changes, temporal overlap, signed coefficients, orthogonal and unit-row controls; no unique source or spike-specific-mode claim. |
| V-C: all 36 matched outcomes | `matched-curves.npz`, `matched-heldout.csv`, `matched-outcomes.csv`, and `main-presentation_manifest.json` | Common forks and fixed horizons; all 33 treatments versus unchanged Adam; q50 retained as preselected despite stronger ResNet q25/q75 endpoints. |
| V-C: displacement mechanism, output reversal, and costs | Same stage arrays and MLP output decomposition; component-cost table and Analysis 065 observation 016 | Local direct bound only; propagated motion and loss unconstrained. Smaller extreme jumps can coexist with more reversal and more cumulative motion. Held-out losses and controller cost remain in main text. |
| S5: historical MLP replication and rate-by-quantile ablation | Preserved Run 286–288 figure and observations, state 448–1152 | Separate windows/calibration; not pooled as seed replication of the new matched matrix. |

All paths in this table are relative to consolidation unless an analysis/run is named. The [preservation manifest](preservation.json) records the exact 406 protected inputs. Original per-figure observations retain the raw-source hashes and generating scripts.

## Adversarial dispositions

- **Motivation before gaps:** computation and training-progress stakes lead the introduction; related work supports concrete questions rather than a claim that prior research ignored representations.
- **Curvature contribution:** explicit path-average identity, pointwise error, sampling contamination, and independently computed HVP reference. No claim to invent secant curvature or obtain the full Hessian.
- **Local versus cumulative interference:** exact all-pair audit replaces an assumed lag-one dominance. Low displacement/path ratio is not called wasted learning.
- **Finite-step inference:** an explicit residual and the 1,325/2,999 high-rate Transformer sign disagreements prevent treating the quadratic formula as a universal nonlinear loss oracle.
- **Adam semantics:** raw-gradient response is a mixed Hessian form along the applied increment. It is not preconditioned sharpness or evidence of preconditioner decoupling.
- **Temporal structure:** energy is squared movement; overlap compares temporal subspaces; coefficient sign uses one fitted spatial pattern. Concentration alone is time-permutation invariant. Amplitude controls are descriptive references, not statistical null tests.
- **Cap semantics:** explicit dimensions, per-output-unit scores, one fixed past-calibrated budget per module, actual stored proposals, retained Adam moments, and unconstrained upstream/interaction terms. The controller needs current probe inputs and local products, not temporal SVD or one altered-network forward per unit.
- **Comparison strength:** report lowest observed mean within the tested grid, not universal superiority. Retain ResNet q50's small PV effect and worse endpoint, every q50 terminal-test increase, unequal effective strengths, and measured controller cost.
- **Replication:** 36 conditions are not 36 independent initializations. Updates, stages, modes, and quantiles do not create seed-level uncertainty. Historical MLP replication is clearly separated.
- **Author-facing scope:** the complete paper is the review entry point; evidence records and this audit are provenance, not additional required reading.

## Literature verification

Primary sources were checked on 14 November 2026. Existing bibliography snapshots remain intact; the complete draft adds four entries and compiles only cited entries. Direct predecessors receive method-level comparisons; the two newer preprints receive only bounded context statements.

| Role | Verified primary source / location | Boundary used in the paper |
|---|---|---|
| Scalar secant | [Barzilai–Borwein, printed p. 142, Eq. 6](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf); [Tan et al., §2, Eq. 6](https://papers.neurips.cc/paper_files/paper/2016/file/c86a7ee3d8ef0b551ed58e354a836f2b-Paper.pdf) | Established reciprocal-BB1/scalar-Hessian interpretation; our path/noise/measurement chain is the positioning. |
| Temporal products | [Chee–Toulis](https://proceedings.mlr.press/v84/chee18a.html), [Pesme et al.](https://proceedings.mlr.press/v119/pesme20a.html), [Baydin et al.](https://arxiv.org/abs/1703.04782) | Stationarity and adaptation are prior uses; sign alone does not diagnose useful learning. |
| Oscillatory descent | [Cohen et al., EoS](https://arxiv.org/abs/2103.00065); [central flows, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/62bf42cc047db5b290e7d5737c1f6a8d-Abstract-Conference.html) | Progress despite oscillation is established; accumulation and functional measurement make our specific connection. |
| Adaptive geometry | [Adaptive Gradient Methods at the Edge of Stability, v2](https://arxiv.org/abs/2207.14484v2); [Bai et al., official ICML 2026 record](https://proceedings.mlr.press/v306/bai26o.html), [§§4–5](https://arxiv.org/html/2506.04805v2) | Preconditioned geometry and second-moment tracking; not a threshold for every raw-gradient diagnostic. |
| Representation-aware control | [Kosson et al., §6/App. B.6](https://arxiv.org/html/2410.23922v1); [AGC, §4/Eq. 3](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf); [FOOF, §4](https://proceedings.mlr.press/v162/benzing22a.html) | Relative layer correction, raw-gradient/weight ratios, and neuron-space descent respectively; none supports a first-representation-aware claim for our cap. |
| Function/representation context | [Catapults](https://proceedings.mlr.press/v235/zhu24h.html); [SVCCA](https://arxiv.org/abs/1706.05806); [CKA](https://proceedings.mlr.press/v97/kornblith19a.html); [loss-landscape visualization](https://arxiv.org/abs/1712.09913) | Credit feature-learning and representation-comparison context; our fitted object is signed consecutive change, not representation similarity or NTK eigenvectors. |
| Recent context | [Terjék, v2](https://arxiv.org/abs/2502.13110v2); [Jiang et al.](https://arxiv.org/abs/2507.12837) | Feature structure and NTK eigenspace evolution near EoS. Metadata/abstract-level support for narrow context; no exhaustive theory comparison claimed. |

## Numerical, mathematical, and publication checks

- [01_prepare.py](01_prepare.py): combines bibliography snapshots plus verified additions; creates Table S1 and supplemental display/table wrappers without changing evidence.
- [02_build.py](02_build.py): full three-pass LaTeX/BibTeX builds, warning collection, PDF hashes, and page renders. Both documents have zero compilation warnings; all fonts are embedded; no unresolved references, placeholders, or out-of-page text spans.
- [03_verify.py](03_verify.py): independently recomputes all 144 main intervention metrics from retained trajectories/evaluations; checks all 144 rounded PDF table cells, continuation counts, nine cap/model effects, motion percentages, MLP output accounting, GD signs, severity precision denominators, and current spectral-control rows.
- **Mathematical checks:** 128 deterministic synthetic smooth-loss/cap cases test path curvature, explicit sampling terms, trapezoidal residual and bounds, radial direct-score bounds, and the full upstream/local decomposition. Maximum identity error is below `2.2e-15`. These verify algebra and implementation; they are not new empirical paper evidence.
- [04_review_sheets.py](04_review_sheets.py): page counts come from the current PDFs, so old scratch renders cannot enter the review. All 14 main and 33 supplemental pages inspected; critical equations, table, algorithms, and updated pages inspected individually. [Hash-bound visual record](visual_review.json).
- **Layout:** nine main figures, three tables, two algorithms, 25 equations, 219-word abstract; 24 supplemental figures and nine tables. The revised larger ResNet event is used consistently. The final pass fixed a one-word column orphan and the augmented proposal-vector notation.
- **Preservation:** every one of the 406 protected files passes SHA-256 comparison, including human notes, the evidence package, tracked active manuscript sources/assets, and the earlier V-C excerpt.
- **Reproduction:** commands and dependencies are in the [draft README](../../draft/README.md). Large source trajectories already in the evidence package remain the scientific source; scratch renders/LaTeX auxiliaries are ignored. No claim of byte-identical PDF rebuilds across TeX installations or creation timestamps.

## Remaining decisions

- **Next:** author review of the complete main PDF; no transfer into `manuscript/draft/` yet.
- **Scientific extensions:** initializations 1/2 for the matched grid, cross-model cap × LR, prospective alarms, independent probes/quiet windows, and matched representation-aware comparators remain conditional on stronger claims. Existing TODOs retain their budgets; none was executed here.
- **Submission:** the decision-letter route restriction already recorded in PLAN and the current venue page/format allowance still require resolution before an actual submission. The present 14-page IEEE-style review layout does not certify TNNLS submission eligibility. No author contact or submission was performed.
