# Literature and positioning map

**Primary-source search: 8 November 2026.** This is a targeted first-pass audit,
not an exhaustive novelty certificate. Version and reading depth matter; recent
preprints are identified separately. Boundaries described as a project comparison
are our inference, not a claim that the cited authors failed at their own task.

## 1. Consequential primary references

| ID | Source, version and inspected location | Established point and implication for this paper |
|---|---|---|
| L01 | Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/pdf/2103.00065), ICLR 2021; main results, Fig.1, Appendix O. | Short-term nonmonotonic loss can coexist with longer-term progress. Credit this foundation; reversals or excursions alone do not establish failed learning. |
| L02 | Cohen et al., [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/pdf/2207.14484), 2022 preprint; abstract and preconditioned-sharpness experiments. | Adaptive stability uses transformed geometry and momentum-dependent conventions. A raw GD Hessian threshold cannot be transplanted into Adam. |
| L03 | Bai et al., [Adaptive Preconditioners Trigger Loss Spikes in Adam](https://arxiv.org/html/2506.04805v2), v2, 25 May 2026; Sections4.1-4.3,5,6.2,7, Appendix D. | Explains preconditioner/gradient decoupling and evaluates lower beta2. Eq.7 scales the Hessian using preconditioning, momentum and bias correction. The threshold is not simply raw `lambda_max(H)>2/eta`. This is the principal Adam-mechanism comparator. |
| L04 | Zhu et al., [Catapults in SGD](https://proceedings.mlr.press/v235/zhu24h.html), ICML 2024; [primary full text](https://arxiv.org/html/2306.04815v3), Sections3.1,3.3,4. | Directly connects low-dimensional output/NTK dynamics, loss spikes, feature learning and generalization. Closest identified conceptual predecessor to the tracing narrative. Our proposed distinction is activation-increment tracing across stages, not first association of spikes and features. |
| L05 | Kosson, Messmer and Jaggi, [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/html/2410.23922v1), NeurIPS 2024; Section6 and Appendices B.4-B.6. | Analyzes representation changes induced by parameter updates and develops relative representation-change corrections. Closest identified predecessor to the cap rationale. The exact cap/control difference still requires a formula-by-formula comparison. |
| L06 | Terjek, [Feature Learning Beyond the Edge of Stability](https://arxiv.org/html/2502.13110v2), v2,17 May 2025; Section2, Definition5, Eq.1, Fig.1, limitations Section3. | Connects activation/gradient structure and feature soft rank, including depthwise scaling for homogeneous MLPs. Eliminates a field-wide claim that stability analysis ignores representation learning. |
| L07 | Jiang, Cohen and Li, [Understanding the Evolution of the Neural Tangent Kernel at the Edge of Stability](https://arxiv.org/html/2507.12837v1), July 2025; Sections3.1-3.2,4, Appendix A.2. | Studies NTK eigenspace/target evolution. Our whole-window fixed SVD is a different measurement and does not prove local axes remain stationary. |
| L08 | Raghu et al., [SVCCA](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf), NeurIPS 2017; Section2 and training-dynamics applications. | SVD-based representation comparison across layers/time is established. Distinguish consecutive motion, signed reversal, event timing, observation controls and intervention use. |
| L09 | Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019; primary proceedings and representation-similarity discussion. | CKA addresses similarity and invariance issues. Raw coordinate motion answers a different question and is parameterization/coordinate dependent. |
| L10 | Li et al., [Visualizing the Loss Landscape of Neural Nets](https://papers.neurips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.pdf), NeurIPS 2018; Section4, Section6/Figs4-6, Appendix B. | Normalized landscape slices demonstrate architecture sensitivity. They do not establish universal flattening from skips, fractality, or basin transitions in our experiments. |
| L11 | Barzilai and Borwein, [Two-Point Step Size Gradient Methods](https://doi.org/10.1093/imanum/8.1.141), 1988, Section2; [primary paper scan](https://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf). | Secant step-size construction predates this work. Reciprocal BB1 equals the proposed kappa proxy under a plain GD step when defined; that equivalence is our algebraic comparison. |
| L12 | Cohen et al., [Understanding Optimization in Deep Learning with Central Flows](https://proceedings.iclr.cc/paper_files/paper/2025/file/62bf42cc047db5b290e7d5737c1f6a8d-Paper-Conference.pdf), ICLR 2025; main formulation, screened for positioning. | Averaged dynamics already separate rapid oscillations from slow progress. Full theorem comparison is not part of this first pass. |

## 2. Two comparisons that can change the contribution claim

**Tracing versus L04/L08/L09.** Prepare a side-by-side method table with measured
object (output residual, representation, or consecutive increment), time
resolution, layer coverage, input/stochastic controls, dimensional reduction,
normalization/invariances, cost, and causal status. Our candidate contribution
combines fixed probes, every-update direction and amplitude, exact temporal
Grams, and graph-aware comparison. Those ingredients being useful does not
establish that the combination is novel. Determine which capability and result
are absent from the nearest verified methods.

**Local caps versus L05.** Compare the exact representation-change quantity,
absolute versus relative scaling, per-unit versus aggregate control, fixed-input
assumption, calibration history, optimizer-state treatment and deployment cost.
The proposed per-unit affine cap may differ meaningfully, but this remains an
unresolved novelty question. A direct compatible baseline is recommended if C3
is presented as a method advance. A task-specific adaptation must be named as
such and cannot be described as an exact reproduction of the original method.

For L03, distinguish the existing late beta2 switch from changing beta2 throughout
training. Its sustained indicator uses neighboring curvature values including
the following time point (Section6.2/Appendix D); its use at the center is
retrospective. The present paper should therefore state its own availability
timeline precisely rather than inherit predictive language from another paper.

## 3. Recent adjacent work and remaining search

| Source | Screen depth and relevance |
|---|---|
| Tang, Chen and Liu, [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/html/2609.18314v2), v2,25 September 2026. | Targeted Sections2-4; preprint. Finite-step geometry and moment timescales are already being studied beyond the quadratic case. |
| Kumar, [Can Representation Learning Decouple from Loss Minimization? Polar Updates Have an Answer](https://arxiv.org/html/2609.36240v1), September 2026. | Targeted Sections5-6; preprint. Muon/teacher-student context, including oscillations and feature learning. Broader technical validation pending. |
| Hofmann, [The Map Behind the Flow](https://arxiv.org/html/2607.04993v1), July 2026. | Targeted framing screen; preprint. Finite-step structured dynamics/representation selection; full technical audit pending. |
| Cattaneo and Shigida, [The Effect of Mini-Batch Noise on the Implicit Bias of Adam](https://arxiv.org/abs/2602.01642), v2,7 May 2026. | Abstract-level screen only. Relevant to interpreting beta choices across batch regimes; no detailed theorem or empirical claim adopted here. |

The further search in P01 should cover representation/feature-motion control,
maximal-update parameterization, natural-gradient/trust-region methods,
layerwise update clipping, and dynamic mode decomposition/proper orthogonal
decomposition of training trajectories. These are search leads, not claims of
equivalence or proven gaps. Citation follow-ups and alternate terminology should
be checked before stating priority or novelty.

## 4. Gap verdicts and writing implications

| Candidate gap | Verdict | Usable form |
|---|---|---|
| Optimization stability ignores learned representations | Does not survive the primary literature. | Credit feature-learning accounts, then identify what localization/intervention question remains. |
| No one has connected low-dimensional dynamics and spikes | Does not survive L04 and related work. | Compare increment-based layerwise tracing with output/NTK analyses. |
| Representation-informed update control is new | Does not survive L05 at that breadth. | Identify the exact rule and experimental distinction, subject to the remaining comparison. |
| A reproducible bridge from cheap trajectory response to architectural motion and controlled intervention | Plausible, still provisional. | State the specific measured capability and tested scope; demonstrate added value against nearest methods. |
| The sign of oscillation alone is insufficient to rank interventions | Supported by local evidence; broad novelty not established. | Present the failed directional controls and successful amplitude allocation as an informative bounded finding. |

The introduction can motivate explanation and intervention without asserting a
new field. If the closest-work audit removes the proposed distinction, revise
the contribution itself or select a discriminating experiment. Adding more
emphatic prose cannot repair an absent gap.

## 5. Submission-format check

The [official TNNLS author-information page](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-information-for-authors)
was not accessible through the research tool in this pass. Article type, current
length limits, supplements, and required declarations remain a submission task;
no policy is inferred from another IEEE journal. The [IEEE reproducibility guide](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)
supports preparation of documented methods, data and code. Preparing a release
does not authorize publishing this repository.
