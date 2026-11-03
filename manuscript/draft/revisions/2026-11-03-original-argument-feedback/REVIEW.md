# Author-feedback revision decisions

## Assessment before editing

The 41 annotations govern this pass. Their original wording is in feedback.md;
TASKS.md records the sequential work. The supplied PDF, original introduction,
Adam section, conclusion and all existing figure/table assets are protected by
SHA-256. A new spectral-overlay figure will use a new filename.

- **C07 - break-even:** retain two contextual sentences about how early learning
  rate and batch size shape later curvature and gradient variability. Remove
  the unused stability equation and its auxiliary notation. The
  [original study](https://arxiv.org/abs/2002.09572) does not establish the
  fixed-state cosine-attenuation claim; our sampling derivation supplies that
  separate argument.
- **C08 - EoS:** distinguish a fixed quadratic's local instability from progress
  on a nonlinear trajectory. [Self-stabilization](https://arxiv.org/abs/2209.15594v2)
  explains curvature feedback, while
  [central flows](https://arxiv.org/abs/2410.24206)
  describe the time-averaged trajectory and oscillation covariance. Cite both
  alongside the original EoS and subsequent analysis.
- **C11 - practice:** organize the reports around recovery and prevention.
  PaLM and OPT support checkpoint recovery and recipe changes; OLMo 2 supports
  loss/gradient monitoring and recipe ablations.
  [Kimi K2](https://arxiv.org/html/2507.20534v1) adds QKClip, which rescales
  query/key weights based on attention logits (it is not gradient clipping).
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437v2) reports no irrecoverable
  spikes or rollbacks, not an absence of every spike. Keep Llama 3's successful
  recipe as context. Omit floating-point implementation detail from this
  argument.
- **C20/C25/C31 - interpretation:** negative consecutive cosine measures
  opposition to preceding parameter changes. The connection to forgetting is
  an interpretation to investigate, not an observed forgetting result.
  [Gradient Episodic Memory](https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
  connects cross-task gradient conflicts to past-task loss. Orthogonal steps
  lack directional reinforcement; their components do not cancel.
  Low normalized displacement quantifies path retention, not learning quality.
- **C28 - noise:** retain the exact nonpositive expected noise interaction and
  finite-population factor. Explain cosine attenuation through normalization
  and shared-signal dilution. The negative unnormalized term by itself cannot
  prove that a negative cosine increases toward zero along an evolving path.
- **C36/C37 - loss:** distinguish a loss-increasing step under the second-order
  approximation from the subsequently defined recovered spike event. Give
  both necessary ingredients, reversal and norm growth, and their joint
  quantitative condition.
- **C40 - spectral comparison:** reuse the 279 retained leading-eigenvalue
  measurements; no Hessian computation or training. MLP coverage is the
  complete 5,000-image objective; ResNet coverage is five fixed normalization
  groups (5,000/50,000 images); Transformer coverage is the first 2,500/59,680
  sequences. The online diagnostic is directional and path averaged; the
  spectrum uses the largest eigenvalue and, for two models, another coverage.
  Compare the regime signatures without claiming eigenvalue calibration.
  In the bottom row align spectral state s to gradient-pair index t=s+2.
  Keep all 31 spectral states per run, including the final point at t=3002.
- **C40/C41 - experimental argument:** retain the evidence for alternating
  backward-lag alignment, including its nonmonotone envelope. Describe
  small-batch attenuation observed from the trained states, while preserving
  the initial-state dependence. Remove the endpoint table from compilation
  and move essential rates into setup prose.

## Comment dispositions

All 41 comments are addressed below. Locations refer to the revised paper.
The mathematical qualifications in the assessment above are retained where
they affect the meaning of the author's interpretation.

| Comment | Location | Completed action |
| --- | --- | --- |
| C01 | II opening | Starts from reinforcement, overshoot and reversal, then distinguishes productive oscillation from instability. |
| C02 | II title | Removed Background. |
| C03 | II-A and references | Consolidated classical citations inside the sentence; checked the PDF for the isolated citation line. Removed redundant printed URLs from eight references while preserving exact sources in BibTeX comments. |
| C04 | II-A | Removed auxiliary perturbation/eigencomponent variables; explains multiplication by 1-eta*lambda directly. |
| C05 | II-A, II-D, III-B, IV-A | Carries the online-cost argument from repeated HVPs to one retained gradient and ordinary reductions. |
| C06 | II-B | Condensed the covariance cost to missing per-example information and its relevance to alignment. |
| C07 | II-C | Retained the hyperparameter/geometry connection after primary-source review; removed the unused break-even equation. |
| C08 | II-C | Added the nonlinear self-stabilization and central-flow explanations of oscillatory convergence. |
| C09 | II-D conclusion | Connects practical norm monitoring to direction, directional curvature and local loss increases. |
| C10 | II-D conclusion | One consolidated transition to the proposed diagnostics. |
| C11 | II-D | Shorter recovery/prevention account using PaLM, OPT, OLMo 2, Kimi K2, Llama 3 and DeepSeek-V3. |
| C12 | III title | Constructive and Destructive Regimes. |
| C13 | III opening | Develops the practical motivation for reading training progress from existing gradients. |
| C14 | III opening | Uses a connected argument from accumulated change to curvature and loss increase. |
| C15 | III decomposition | Removed the redundant eta-squared step-length phrase. |
| C16 | III decomposition | Replaced unexplained first-order language with local steepness. |
| C17 | III decomposition | Explains individual nonnegative squared-norm contributions and steepest descent. |
| C18 | III decomposition | Emphasizes destructive interference at the first cross-step inner product. |
| C19 | III decomposition | Explicitly factors each dot product into two norms and a cosine. |
| C20 | III regime definition | Connects both regimes directly to SGD's accumulated update equation and parameter change. |
| C21 | III-A opening | States the curvature/regime connection before the derivation. |
| C22 | III-A definitions | Defines curvature and sampling terms before interpreting them. |
| C23 | III-A, Eq. (14) | Boxed the central cosine/curvature relation. |
| C24 | III-A | Explains zero full-batch noise, positive norm ratio and the regime boundary at one. |
| C25 | III-A interpretation | Gives parameter-overwriting and optimization interpretations; cites GEM and separates behavioral forgetting from parameter geometry; removes alpha notation. |
| C26 | III-B | Added a dedicated Hessian-free estimation subsection with O(p) cost, one stored gradient, and the batch-size bridge. |
| C27 | III-B and Appendix A | Specifies finite-step projected change, smooth-GD path average, local approximation and SGD sampling term. |
| C28 | III-C | Retains nonpositive noise expectation, finite-population factor and fixed-state attenuation; removes the general two-state display and noise-energy wording. |
| C29 | III-D opening | Motivates retention through traveled distance versus net parameter change. |
| C30 | III-D | Removed the second R-squared display; explains all-pair contributions in prose. |
| C31 | III-D | Added 100-step/10,000-step orthogonal examples: 10%/1% retention; distinguishes lack of reinforcement from cancellation. |
| C32 | III-E title | Conditions for Loss Decrease and Spike Formation. |
| C33 | III-E and Appendix A | Capital Delta for every active loss-difference expression. |
| C34 | III-E opening | Opens with when a destructive regime produces a loss increase. |
| C35 | III-E | Explains reversal jointly with gradient growth before introducing the curvature equivalent. |
| C36 | III-E, Eq. (21) | Boxed both equivalent conditions; states both necessary ingredients and two numerical angle examples. |
| C37 | III-E closing | Concise curvature form; distinguishes thresholds one/two and a local increase from a recovered spike. |
| C38 | IV opening | States the experimental goals and matched objectives, initializations, rates and measurement scope. |
| C39 | IV title | Centers constructive and destructive regimes. |
| C40 | IV-A-D and Figs. 1-4 | (a) Added all 279 retained spectra with exact coverage/indexing; (b) explained loss, reversal, norms and retention; (c) interpreted lag alternation and its nonmonotone envelope; (d) explained trained-state attenuation and initial-state dependence; (e) removed the endpoint table from compilation and retained its source. |
| C41 | IV and document layout | Revised every subsection; removed the forced SGD page break; queued controlled figures together on pp. 5-7 before Adam; removed the obsolete reference-column break and compacted the new figure. |

## Verification and final layout

The resulting paper has 13 pages, eight figures and one table. All 41
annotations are mapped above and in TASKS.md. The controlled figures occupy
pages 5-7, and Adam starts on page 8. Figure 1 has the original loss and
smoothed diagnostic values plus the existing unsmoothed spectra; the larger
spectral values are shown without axis clipping.

The verification script passes 240 added quadratic loss-condition cases,
186 retained quadratic/noise checks and 508 retained geometry/path-average
checks. It reconstructs the existing GD/SGD plotted reductions and independently
compares all new Figure 1 series to raw event/spectral records (53,838 plotted
values, including 279 spectral measurements). Twenty-nine protected files match
their original hashes, including the annotated PDF and prior revision.

All 13 final pages were rendered with Poppler and inspected. The boxed
relations, equation indices, figure legends, loss tick format, table, and
reference ending are readable. Compilation has no overfull boxes, undefined
references or multiply defined labels. The obsolete temporal-accounting pointer
was removed with the rewritten paragraph; no removed appendix content was
restored. The two appendices remain limited to derivations and essential
measurement conventions.

This pass uses only retained evidence. No training, HVP computation,
intervention, remote resource, or scientific-input change was performed.
