# Literature and positioning within the current plan

- **Updated:** 8 November 2026; aligned with the four contributions in [PLAN.md](../PLAN.md).
- **Scope:** targeted primary-source review, with reading depth recorded below. Specific differences are not claims that prior work failed at its own objective.
- **Bibliography:** [current snapshot](02-related-work/bibliography-current.bib) and [eight verified additions](02-related-work/references-to-add.bib); active manuscript bibliography unchanged.
- **Freshness:** title/author/version metadata for the four recent arXiv records was rechecked during packaging. Earlier technical reading depths are retained, not upgraded to new full-proof audits.

## 1. Gap-to-contribution alignment

| Contribution | Predecessor role | Specific response and boundary |
|---|---|---|
| 1. Local reversal and accumulation | EoS, stationarity diagnostics and hypergradients already use oscillation or gradient products. | Joint interpretation of local sign, all-pair accumulation and measured progress; not first use of inner products. |
| 2. Inexpensive directional curvature | BB supplies the reciprocal secant; Tan explicitly interprets scalar Hessian approximation; spectral studies measure different quantities. | Path-average meaning, pointwise error and explicit sampling term, checked against independent MLP directional HVPs. |
| 3. Reversal, amplification and loss | EoS/central flows explain oscillatory progress; Bai studies adaptive preconditioner dynamics. | GD finite-step criterion with residual checks, optimizer-dependent response and exhaustive Adam timing/severity characterization. No universal alarm. |
| 4. Functional tracing and control | Catapults/feature-learning work, SVCCA/CKA, Kosson, AGC and FOOF cover nearby objects. | Consecutive-motion localization and per-unit, past-calibrated direct affine cap, with bounded case studies and negative controls. No first representation-aware optimizer or untested superiority claim. |

## 2. Primary references for the original core

IDs remain stable for other planning documents; the order places the original
core before its Section V extension.

| ID | Source and reading depth | Established result and role |
|---|---|---|
| L13 | Chee and Toulis, [Convergence diagnostics for stochastic gradient descent with constant learning rate](https://proceedings.mlr.press/v84/chee18a.html), AISTATS 2018; [PDF](https://proceedings.mlr.press/v84/chee18a/chee18a.pdf), targeted Sections 1.1,2.1/Algorithm 1 and 3, PDF pp.2-4. | Running successive-gradient products underpin a stationarity diagnostic with noise-dependent analysis. Close antecedent for temporal-gradient interpretation; distinguish our regime characterization from declaring convergence. Full proofs/supplement not audited. |
| L14 | Pesme, Dieuleveut and Flammarion, [On Convergence-Diagnostic based Step Sizes for Stochastic Gradient Descent](https://proceedings.mlr.press/v119/pesme20a.html), ICML 2020; [PDF](https://proceedings.mlr.press/v119/pesme20a/pesme20a.pdf), targeted Sections 4.1/Proposition 9,4.2/Proposition 13 and 5, PDF pp.4-6. | Separates full-gradient and noise contributions to the expected product; noise dominates its negative stationary expectation under their assumptions. Shows inadequate statistical detection in a quadratic setting and proposes a distance criterion. Essential to accumulation and sampling and the distinction between an identity, expectation and detector. Full proofs unaudited. |
| L15 | Baydin et al., [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782), ICLR 2018, v3; [PDF](https://arxiv.org/pdf/1703.04782), targeted Section 2 Eqs.2-3, Section 2.1 Eqs.6-8 and algorithms, PDF pp.2-4. | Consecutive-gradient products yield the GD learning-rate hypergradient; general optimizers use their own update derivative. Adam uses its adaptive direction. Low-cost previous-gradient information is established; our distinction is the response interpretation and connected explanatory tests. Full convergence analysis not reviewed. |
| L11 | Barzilai and Borwein, [Two-Point Step Size Gradient Methods](https://doi.org/10.1093/imanum/8.1.141), 1988; Section 2, [primary scan](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf). | Classical secant step-size construction. Reciprocal BB1 equals the proxy under a plain GD step when defined; that equality is our algebraic comparison. Credit it while specifying the regime/loss/noise interpretation and validation. |
| L01 | Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/pdf/2103.00065), ICLR 2021; main results, Fig. 1 and Appendix O. | Short-term nonmonotonic loss coexists with longer-term progress. Develop how this appears in gradient geometry and accumulation, rather than claiming oscillatory progress as a first discovery. |
| L12 | Cohen et al., [Understanding Optimization in Deep Learning with Central Flows](https://proceedings.iclr.cc/paper_files/paper/2025/file/62bf42cc047db5b290e7d5737c1f6a8d-Paper-Conference.pdf), ICLR 2025; main formulation screened. | Averaged dynamics separate rapid oscillations and slower progress. Compare with the manuscript's temporal observables; full theorem-level comparison remains outside the audited theorem-level scope. |
| L10 | Li et al., [Visualizing the Loss Landscape of Neural Nets](https://papers.neurips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf), NeurIPS 2018; Sections 4/6, Figs.4-6 and Appendix B. | Normalized slices show architecture-dependent geometry, supporting the architecture/learning motivation. They do not establish universal skip-induced flattening, fractality or basin transitions in our trajectories. |
| L16 | Yunwen Lei, [Towards Better Generalization Bounds of Stochastic Optimization for Nonconvex Learning](https://leiyw.github.io/paper/TPAMI2025.pdf), author-hosted TPAMI paper; abstract and Sections I-IV inspected for positioning. | High-probability population-gradient/generalization bounds and second-moment uniform convergence address statistical learning guarantees. This is a distinct target from temporal regime interpretation and measured finite-step dynamics; the foundational contributions do not replace such bounds or imply generalization guarantees. Full theorem/proof audit not performed. |
| L17 | Shaojie Li and Yong Liu, [High Probability Guarantees for Nonconvex Stochastic Gradient Descent with Heavy Tails](https://proceedings.mlr.press/v162/li22q.html), ICML 2022; primary proceedings abstract screened. | Studies optimization/generalization under sub-Weibull noise and additional assumptions, including a clipping analysis that removes a bounded-gradient assumption. Cite for stochastic assumptions and clipping context; our completed empirical clipping contrasts answer a different, finite-window intervention question. Detailed theorem comparison remains unreviewed. |

## 3. Primary references for the Adam and representation extension

| ID | Source and inspected location | Established point and positive distinction to examine |
|---|---|---|
| L02 | Cohen et al., [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/pdf/2207.14484), 2022 preprint; abstract and preconditioned-sharpness experiments. | Adaptive stability depends on transformed geometry and momentum conventions. The Adam extension specifies the same raw-gradient observable along an adaptive path. |
| L03 | Bai et al., [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://proceedings.mlr.press/v306/bai26o.html), ICML 2026; [inspected v2](https://arxiv.org/html/2506.04805v2), 25 May 2026, Sections 4.1-4.3,5,6.2, 7 and Appendix D. | Preconditioner/gradient decoupling, transformed sharpness and lower-beta2 experiments are the principal Adam comparison. Eq. 7 includes preconditioning, momentum and bias correction; its 2/eta boundary is not a raw-Hessian threshold. |
| L04 | Zhu et al., [Catapults in SGD](https://proceedings.mlr.press/v235/zhu24h.html), ICML 2024; [full text](https://arxiv.org/html/2306.04815v3), Sections 3.1,3.3 and 4. | Connects low-dimensional output/NTK dynamics, spikes and feature learning. Compare consecutive activation-increment tracing across stages with direction and amplitude. |
| L05 | Kosson, Messmer and Jaggi, [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/html/2410.23922v1), NeurIPS 2024; Section 6 and Appendices B.4-B.6. | Update-induced representation change and relative-change corrections are established. Closest identified antecedent to the cap rationale; compare the exact control with our per-unit, past-calibrated local affine cap. |
| L06 | Terjek, [Feature Learning Beyond the Edge of Stability](https://arxiv.org/html/2502.13110v2), 17 May 2025; Section 2, Definition 5, Eq. 1, Fig. 1 and Section 3. | Activation/gradient structure, feature soft rank and homogeneous-MLP scaling connect learning and sharpness. The tracing/intervention contribution adds measured motion localization and intervention consequences in the selected setups. |
| L07 | Jiang, Cohen and Li, [Understanding the Evolution of the Neural Tangent Kernel at the Edge of Stability](https://arxiv.org/html/2507.12837v1), July 2025; Sections 3.1-3.2,4 and Appendix A.2. | Studies NTK eigenspace/target evolution. Its object differs from temporal SVD of representation increments; the axes must not be identified with each other. |
| L08 | Raghu et al., [SVCCA](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf), NeurIPS 2017; Section 2 and training-dynamics applications. | Establishes SVD/CCA representation comparison. The tracing comparison concerns signed consecutive motion, event timing, stage localization and intervention use. |
| L09 | Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019; proceedings and similarity discussion. | Similarity/invariance questions differ from coordinate-dependent displacement. Explain why displacement answers the proposed update-response question. |

The current plan uses these three comparisons:

1. **Core to Adam:** distinguish raw-gradient response, GD/Adam hypergradients,
   and L03's transformed-curvature predictor, including cost and information
   availability. L03 Section 6.2/Appendix D uses a following curvature value in
   its centered sustained indicator. The completed V-A audit supplies time-correct descriptive evidence;
   validated forecasting remains conditional.
2. **Tracing:** compare object, resolution, layer coverage, fixed-input/stochastic
   controls, reduction, cost and causal status against L04/L08/L09. State the
   capability and findings delivered by 063.
3. **Cap:** compare absolute/relative displacement, unit/aggregate control,
   frozen-input assumption, calibration, Adam-state treatment and cost with L05.
   Identify a faithful baseline only if a method-superiority claim needs it;
   task-specific adaptations must be identified as such.

Analysis 059's late beta2 switch retains accumulated state and changes native
bias normalization. It is useful evidence about that intervention, rather than
a general refutation of lower beta2 from initialization.

## 4. Recent work and bounded follow-up

| Source | Screen depth and relevance |
|---|---|
| Tang, Chen and Liu, [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/html/2609.18314v2), 25 September 2026. | Targeted Sections 2-4; preprint. Finite-step geometry and moment timescales beyond quadratics. |
| Kumar, [Can Representation Learning Decouple from Loss Minimization? Polar Updates Have an Answer](https://arxiv.org/html/2609.36240v1), September 2026. | Targeted Sections 5-6; preprint. Muon/teacher-student oscillations and feature learning; broader technical validation pending. |
| Hofmann, [The Map Behind the Flow](https://arxiv.org/html/2607.04993v1), July 2026. | Framing screen; preprint. Structured finite-step dynamics and representation selection; full technical audit pending. |
| Cattaneo and Shigida, [The Effect of Mini-Batch Noise on the Implicit Bias of Adam](https://arxiv.org/abs/2602.01642), v2, 7 May 2026. | Abstract screen only; relevant beta/batch interaction. No detailed theorem or empirical claim adopted. |

Follow references and alternate terms where they can change a specific contribution
claim: stochastic approximation and temporal-gradient diagnostics for the core;
feature-motion control, layerwise clipping and dynamical decompositions for V.
Do not turn this into an unbounded survey or a new paper structure. L13-L15
primary pages and linked PDFs, and the L03 proceedings record, opened successfully
in the earlier review. Earlier sources retain their recorded versions and depths.

## 5. Formula-level intervention audit, 8 November 2026

This addendum supports the current single PLAN; it does not introduce another
required reading document. Primary PDFs were opened and the specified formulas
checked during the adversarial review.

| Source / inspected location | Established method | Exact distinction for V-C |
|---|---|---|
| Brock et al., [High-Performance Large-Scale Image Recognition Without Normalization](https://proceedings.mlr.press/v139/brock21a/brock21a.pdf#page=4), ICML 2021, Section 4 / Eq. (3) | AGC clips each output unit's raw gradient using its gradient-to-weight norm ratio, with a weight-norm floor. | The local-displacement cap scales the stored Adam parameter increment using current probe-input geometry. Unit-wise clipping itself is established. No superiority to AGC has been tested here. |
| Benzing, [Gradient Descent on Neurons and its Link to Approximate Second-Order Optimization](https://proceedings.mlr.press/v162/benzing22a/benzing22a.pdf#page=6), ICML 2022, Section 4 / Eqs. (5–6) | FOOF solves a regularized least-squares neuron-change objective involving the inverse input second-moment factor. | Our rule evaluates the local response of an existing Adam proposal and radially rescales each unit; it does not invert that factor or solve for a neuron-gradient descent direction. Input-dependent functional geometry is established prior art. |

Together with L05, these narrow the contribution to a specified direct,
per-unit, absolute movement cap with past calibration and controlled empirical
tests. A claim of being the first representation-aware optimizer, or of
superiority over these methods, is not supported. The main plan now states
this comparison and the corresponding conditional experiment requirement.

## 6. Submission route and format

The supplied [TNNLS decision letter](appendices/submission/decision-letter.pdf)
prohibits submission of a revised version to TNNLS. PLAN closure step 5 must resolve the intended
route against that actual letter before submission; ordinary revised
resubmission cannot be assumed. This records a venue-specific issue without
changing scientific direction or selecting another venue.

The [official TNNLS author-information page](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-information-for-authors)
was inaccessible in the initial research-tool pass. Current article type,
length, supplement and declaration requirements remain to be checked through
official sources. The [IEEE reproducibility guide](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
is a packaging reference. Preparation does not authorize release, editorial
contact, pushing or submission.
