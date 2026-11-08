# Literature and positioning within the original argument

**Iteration 2 - primary-source review through 8 November 2026.** This map follows
the fixed six sections and C1-C8 in [PLAN.md](../PLAN.md). It supports the author's
constructive/destructive, curvature, inexpensive-proxy, loss and sampling
argument; Adam, tracing and interventions develop it within Section V. This is
a targeted comparison, not an exhaustive priority determination. A boundary
identified for our question is our comparison, not a claim that another paper
failed at its own objective. Reading depth is stated below.

## 1. Affirmative positioning of the complete argument

Existing work explains oscillatory optimization, uses successive-gradient
products for adaptation and stationarity diagnostics, and studies feature
learning near instability. This paper asks how reinforcement or reversal of
learning updates relates to magnitude, curvature, sampling, accumulated
displacement and loss change. Its contribution is the connected formulation
and empirical explanatory tests: what a regime means, what the inexpensive
observable measures, and which consequences follow in the measured learning
problems. Its value does not depend on inventing every algebraic ingredient.

| Component / placement | Cited predecessor and boundary for our question | Our contribution and existing support | Comparison still to finish |
|---|---|---|---|
| C1: regime characterization, III | L13-L15 already use successive-gradient products. Stationarity tests and LR updates do not by themselves supply the paper's joint account of adjacent interference, accumulated motion and loss. | Formulate constructive/destructive behavior from SGD updates and distinguish local opposition from total motion. Exact decomposition and 024/035/056 test this distinction. | Compare normalization, temporal averaging, assumptions and purpose; do not claim first use of the product. |
| C2: curvature and cheap proxy, III-A/B | L11 supplies the classical secant antecedent; L01 characterizes spectral sharpness. Neither correspondence alone validates the learning interpretation or neural measurements. | Relate alignment and norm ratio to projected gradient change, with a path-average identity and approximation bound. Analyses 017/018/024 and 027 provide existing calibration and derivative comparisons. | Credit BB equivalence; separate direction/path from spectrum and computational order from measured overhead. |
| C3: reversal, amplification and loss, III-E/IV/V | L01/L12 establish oscillation with progress; L14 separates sign expectations from effective detection. | Distinguish reversal from amplification sufficient for a completed-step quadratic loss rise; test finite-step accuracy and neural counterexamples. Analyses 024, 055/057 and 062/065 support the argument. | Attach each boundary to its objective, optimizer and clock; validate the author's pre-update warning objective. |
| C4: noise and batch effects, III-C/IV | L13/L14 already analyze noise contributions to successive-gradient products. | Connect the noise relation to fixed-state sampling, moving sampled paths and full-objective progress in 020/027/037. Analysis 027 includes 1,188 matched HVP pairs. | Compare assumptions with L14 and select concise existing validation; do not infer universal noise-induced stabilization. |
| C5: emergence and accumulation, IV | L01/L12 supply the oscillatory-progress setting. | Characterize the framework across GD rates, exact backward lags and minibatches.024/056 separate adjacent cancellation from positive all-pair interaction; 037 extends batch effects across three model/data cases. | Explain the retained figures without equating path retention with knowledge or generalization. |
| C6: adaptive trajectories, V opening/A | L02/L03 examine adaptive geometry; L15's Adam hypergradient is algorithm-specific. | Keep one raw-gradient computation with its mixed gradient/Adam-direction interpretation.036/055/057 characterize excursions, ordinary periods and decision-time availability. | Compare projected quantities and timing; do not equate this observable with an Adam hypergradient or transplant a GD threshold. |
| C7: tracing, algorithm and findings, V-B | L04/L06/L07 connect instability and feature learning; L08/L09 establish representation comparison. | Follow consecutive functional motion through the architecture on fixed probes, measuring direction, amplitude and temporal structure. Analysis 063 formalizes the method developed in 061/062. | Compare object, event resolution, layer coverage and controls; window concentration does not establish a stationary instantaneous axis or unique cause. |
| C8: informed intervention, V-C | L05 studies update-induced representation change; L03 motivates beta2 changes. | Test local affine-displacement allocation against unchanged Adam and completed controls. Analyses 064/065 include cross-architecture effects, MLP replication and the 54-condition cap/LR grid. | Finish the formula-level L05 comparison and only the missing matched contrasts required by the selected claim; retain held-out tradeoffs. |

The introduction can therefore argue that this framework makes relationships
between learning updates and stability measurable, then follows their
consequences from GD and sampling to Adam's internal motion and interventions.
The [argument map](ARGUMENT_MAP.md) develops that affirmative claim-to-evidence
chain. Familiar mathematical ingredients do not remove the need for these
explanatory relationships and empirical tests.

## 2. Primary references for the original core

IDs remain stable for other planning documents; the order places the original
core before its Section V extension.

| ID | Source and reading depth | Established result and role |
|---|---|---|
| L13 | Chee and Toulis, [Convergence diagnostics for stochastic gradient descent with constant learning rate](https://proceedings.mlr.press/v84/chee18a.html), AISTATS 2018; [PDF](https://proceedings.mlr.press/v84/chee18a/chee18a.pdf), targeted Sections 1.1,2.1/Algorithm 1 and 3, PDF pp.2-4. | Running successive-gradient products underpin a stationarity diagnostic with noise-dependent analysis. Close antecedent for temporal-gradient interpretation; distinguish our regime characterization from declaring convergence. Full proofs/supplement not audited. |
| L14 | Pesme, Dieuleveut and Flammarion, [On Convergence-Diagnostic based Step Sizes for Stochastic Gradient Descent](https://proceedings.mlr.press/v119/pesme20a.html), ICML 2020; [PDF](https://proceedings.mlr.press/v119/pesme20a/pesme20a.pdf), targeted Sections 4.1/Proposition 9,4.2/Proposition 13 and 5, PDF pp.4-6. | Separates full-gradient and noise contributions to the expected product; noise dominates its negative stationary expectation under their assumptions. Shows inadequate statistical detection in a quadratic setting and proposes a distance criterion. Essential to C1/C4 and the distinction between an identity, expectation and detector. Full proofs unaudited. |
| L15 | Baydin et al., [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782), ICLR 2018, v3; [PDF](https://arxiv.org/pdf/1703.04782), targeted Section 2 Eqs.2-3, Section 2.1 Eqs.6-8 and algorithms, PDF pp.2-4. | Consecutive-gradient products yield the GD learning-rate hypergradient; general optimizers use their own update derivative. Adam uses its adaptive direction. Low-cost previous-gradient information is established; our distinction is the response interpretation and connected explanatory tests. Full convergence analysis not reviewed. |
| L11 | Barzilai and Borwein, [Two-Point Step Size Gradient Methods](https://doi.org/10.1093/imanum/8.1.141), 1988; Section 2, [primary scan](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf). | Classical secant step-size construction. Reciprocal BB1 equals the proxy under a plain GD step when defined; that equality is our algebraic comparison. Credit it while specifying the regime/loss/noise interpretation and validation. |
| L01 | Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/pdf/2103.00065), ICLR 2021; main results, Fig. 1 and Appendix O. | Short-term nonmonotonic loss coexists with longer-term progress. Develop how this appears in gradient geometry and accumulation, rather than claiming oscillatory progress as a first discovery. |
| L12 | Cohen et al., [Understanding Optimization in Deep Learning with Central Flows](https://proceedings.iclr.cc/paper_files/paper/2025/file/62bf42cc047db5b290e7d5737c1f6a8d-Paper-Conference.pdf), ICLR 2025; main formulation screened. | Averaged dynamics separate rapid oscillations and slower progress. Compare with the manuscript's temporal observables; full theorem-level comparison remains P01 work. |
| L10 | Li et al., [Visualizing the Loss Landscape of Neural Nets](https://papers.neurips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf), NeurIPS 2018; Sections 4/6, Figs.4-6 and Appendix B. | Normalized slices show architecture-dependent geometry, supporting the architecture/learning motivation. They do not establish universal skip-induced flattening, fractality or basin transitions in our trajectories. |
| L16 | Yunwen Lei, [Towards Better Generalization Bounds of Stochastic Optimization for Nonconvex Learning](https://leiyw.github.io/paper/TPAMI2025.pdf), author-hosted TPAMI paper; abstract and SectionsI-IV inspected for positioning. | High-probability population-gradient/generalization bounds and second-moment uniform convergence address statistical learning guarantees. This is a distinct target from temporal regime interpretation and measured finite-step dynamics; C1-C5 do not replace such bounds or imply generalization guarantees. Full theorem/proof audit not performed. |
| L17 | Shaojie Li and Yong Liu, [High Probability Guarantees for Nonconvex Stochastic Gradient Descent with Heavy Tails](https://proceedings.mlr.press/v162/li22q.html), ICML 2022; primary proceedings abstract screened. | Studies optimization/generalization under sub-Weibull noise and additional assumptions, including a clipping analysis that removes a bounded-gradient assumption. Cite for stochastic assumptions and clipping context; our completed empirical clipping contrasts answer a different, finite-window intervention question. Detailed theorem comparison remains unreviewed. |

## 3. Primary references for the Adam and representation extension

| ID | Source and inspected location | Established point and positive distinction to examine |
|---|---|---|
| L02 | Cohen et al., [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/pdf/2207.14484), 2022 preprint; abstract and preconditioned-sharpness experiments. | Adaptive stability depends on transformed geometry and momentum conventions. C6 specifies the same raw-gradient observable along an adaptive path. |
| L03 | Bai et al., [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://proceedings.mlr.press/v306/bai26o.html), ICML 2026; [inspected v2](https://arxiv.org/html/2506.04805v2), 25 May 2026, Sections 4.1-4.3,5,6.2, 7 and Appendix D. | Preconditioner/gradient decoupling, transformed sharpness and lower-beta2 experiments are the principal Adam comparison. Eq. 7 includes preconditioning, momentum and bias correction; its 2/eta boundary is not a raw-Hessian threshold. |
| L04 | Zhu et al., [Catapults in SGD](https://proceedings.mlr.press/v235/zhu24h.html), ICML 2024; [full text](https://arxiv.org/html/2306.04815v3), Sections 3.1,3.3 and 4. | Connects low-dimensional output/NTK dynamics, spikes and feature learning. Compare consecutive activation-increment tracing across stages with direction and amplitude. |
| L05 | Kosson, Messmer and Jaggi, [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/html/2410.23922v1), NeurIPS 2024; Section 6 and Appendices B.4-B.6. | Update-induced representation change and relative-change corrections are established. Closest identified antecedent to the cap rationale; compare the exact control with our per-unit, past-calibrated local affine cap. |
| L06 | Terjek, [Feature Learning Beyond the Edge of Stability](https://arxiv.org/html/2502.13110v2), 17 May 2025; Section 2, Definition 5, Eq. 1, Fig. 1 and Section 3. | Activation/gradient structure, feature soft rank and homogeneous-MLP scaling connect learning and sharpness. C7/C8 add measured motion localization and intervention consequences in the selected setups. |
| L07 | Jiang, Cohen and Li, [Understanding the Evolution of the Neural Tangent Kernel at the Edge of Stability](https://arxiv.org/html/2507.12837v1), July 2025; Sections 3.1-3.2,4 and Appendix A.2. | Studies NTK eigenspace/target evolution. Its object differs from temporal SVD of representation increments; the axes must not be identified with each other. |
| L08 | Raghu et al., [SVCCA](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf), NeurIPS 2017; Section 2 and training-dynamics applications. | Establishes SVD/CCA representation comparison. The tracing comparison concerns signed consecutive motion, event timing, stage localization and intervention use. |
| L09 | Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML2019; proceedings and similarity discussion. | Similarity/invariance questions differ from coordinate-dependent displacement. Explain why displacement answers the proposed update-response question. |

Three concrete comparisons complete this part of P01:

1. **Core to Adam:** distinguish raw-gradient response, GD/Adam hypergradients,
   and L03's transformed-curvature predictor, including cost and information
   availability. L03 Section 6.2/Appendix D uses a following curvature value in
   its centered sustained indicator. Our pre-update objective needs its own
   time-correct validation.
2. **Tracing:** compare object, resolution, layer coverage, fixed-input/stochastic
   controls, reduction, cost and causal status against L04/L08/L09. State the
   capability and findings delivered by 063.
3. **Cap:** compare absolute/relative displacement, unit/aggregate control,
   frozen-input assumption, calibration, Adam-state treatment and cost with L05.
   Identify a faithful baseline only if the intended C8 claim needs it;
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

Follow references and alternate terms where they can change a specific C1-C8
claim: stochastic approximation and temporal-gradient diagnostics for the core;
feature-motion control, layerwise clipping and dynamical decompositions for V.
Do not turn this into an unbounded survey or a new paper structure. L13-L15
primary pages and linked PDFs, and the L03 proceedings record, opened successfully
in this iteration. Earlier sources retain their recorded versions and depths.

## 5. Submission route and format

The supplied [TNNLS decision letter](../../draft/previous-submission/rejection_email.pdf)
prohibits submission of a revised version to TNNLS. W02 must resolve the intended
route against that actual letter before submission; ordinary revised
resubmission cannot be assumed. This records a venue-specific issue without
changing scientific direction or selecting another venue.

The [official TNNLS author-information page](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-information-for-authors)
was inaccessible in the initial research-tool pass. Current article type,
length, supplement and declaration requirements remain to be checked through
official sources. The [IEEE reproducibility guide](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
is a packaging reference. Preparation does not authorize release, editorial
contact, pushing or submission.
