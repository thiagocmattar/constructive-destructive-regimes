# Round 2: final framing and human-reader review

Date: 5 November 2026.

Reviewed artifact: `manuscript/draft/revisions/2026-11-05-submission-synthesis/revised-manuscript.pdf`.

- Pages: 18.
- SHA-256: `d078731c253ffed9da6d0120795beb205a40bc00b0a785cf1419762e65e1d44f`.
- Title: *Constructive and Destructive Training Regimes: Diagnosing Gradient Interaction*.
- Review scope: final narrative, contribution visibility, evidence relevance and qualification; current active sources and extracted PDF text. Visually inspected pages 1, 7, 13 and 14 using the current rendered pages. This is a focused reader review, not the team's separate all-page layout audit or a new raw-artifact verification.

## Verdict

**No remaining actionable framing or interpretation blocker was found for the chosen diagnostic scope.** The manuscript is ready for the authors' scientific judgment and final submission preparation after the team's numerical, build and complete-layout checks. This verdict concerns whether the present paper makes a coherent, supported argument; it does not predict journal acceptance.

The paper now has a positive contribution that a reader can identify: locate cancellation within pairs, distinguish it from residual accumulation and total interaction, identify the optimizer vector being observed, and evaluate the resulting motion against a declared objective and horizon. The empirical sequence demonstrates why each part is needed. It no longer depends on a new stability theorem, spike predictor, unvalidated curvature approximation, or new intervention method.

## 1. Round-1 issue disposition

| ID | Requested repair | Final status |
| --- | --- | --- |
| R1-A | Show that clipping the optimizer input can increase the applied step | Resolved. Page 13 reports the MLP mean and maximum actual-step norms increasing under clipping, alongside the contrasting decrease in ResNet maximum step norm. The explanation identifies the capped input and optimizer transformation separately. |
| R1-B | Distinguish cumulative Pythia raw-gradient accounting from actual AdamW displacement | Resolved. Page 13 explicitly states that the raw cumulative sum is not AdamW parameter displacement before reporting the all-pair and lag-one results. |
| R1-C | Name the vector/norm in the abstract and conclusion | Resolved. The abstract now refers to more positive raw-gradient alignment or input clipping; the conclusion refers to clipping the optimizer input. Neither silently claims smaller raw or applied norms. |
| R1-D | Explain what paired factors add beyond a near-saturated negative cosine | Resolved. The GD discussion explicitly identifies that the Transformer cancels the largest fraction inside pairs among the three selected paths but retains the largest fraction overall through more coherent residual accumulation. |
| R1-E | Address probabilistic/norm-control literature from the original review | Resolved. Related work cites Li and Liu and Lei and distinguishes their guarantees from the current realized-path identities. |

The optional round-1 suggestion to move some Pythia recipe details into an appendix is not necessary for correctness. The current presentation remains readable and keeps the intervention contracts available next to the evidence.

## 2. Central contribution visibility

The shortened title names the measured object and avoids the previous unsupported promise of general SGD stability. The abstract gives the three diagnostic levels, the all-lag and paired accounting, the GD/SGD/Adam distinctions, and the existing intervention implications. The first-page introduction states the research question explicitly and credits classical gradient-product and secant work before claiming an advance.

The pair-factor discussion on pages 7--8 is the clearest specific addition to a cosine-only account. The three selected trajectories have similar strong adjacent opposition, yet their residual coherence differs substantially. The new comparison sentence makes that added information explicit without equating larger residual coherence with a causal improvement in loss.

The Adam section uses a distinct, answerable question: whether observed loss rises accompany directions that point uphill locally, or finite steps that increase loss despite a descending tangent. The 794-step result answers that question in its recovered-event sample. Its relevance does not depend on successfully forecasting another event. The paper preserves the empirical distinction between a rise interval and every individual loss change inside it.

The implications section now completes the argument. Its diagnostic sequence names `C_T`, `D_T`, `Q_T`, current-gradient/applied-increment alignment, and common-objective loss. A reader can understand which quantity to inspect for each question. This is more useful than a final list of caveats.

## 3. Evidence relevance and source scope

The selected Pythia schedule cases test the relation between raw alignment and full-validation outcome under a complete schedule change. The prose acknowledges that learning rate also scales AdamW decay. It does not attribute the loss effect uniquely to alignment.

The coordinate-scaling intervention is correctly described as matching the native parent's adaptive-component length at corresponding updates, while the branch updates its own moments and weights. The revised text also preserves the original 712-update learning-rate clock for the 350-update parents. The fixed probe is explicitly disjoint from the union of the three 350-update training prefixes in this intervention study. This avoids implying exclusion from every training sequence in the separate full-pass experiment.

The clipping comparison provides the strongest practical illustration of optimizer-stream semantics: a bounded input norm does not impose the same constraint on applied parameter motion. The MLP actual-step increase and the opposite ResNet response make this an observed diagnostic distinction rather than merely an abstract property of Adam. The subsequent loss summaries establish horizon dependence without declaring a globally preferred clipping threshold.

Table II remains a collection of within-study comparisons. It specifies different objectives/windows, reports the scaling-ablation geometric-mean reduction, and explicitly disclaims pooling. The sign conventions in the caption allow readers to compare each row with its own control. The table does not turn heterogeneous examples into independent confirmations of one common causal mechanism.

The evidence limits remain appropriately attached: one initialization in the main full-batch studies, three blocks in specified SGD/Pythia cohorts, dependent recovered events, selected intervention windows, fixed parameterization, and no measured overhead advantage. The appendix connection to Bai is presented as an identity under assumptions, with neural calibration absent. No excluded claim creates a hidden requirement to execute the November 5 experiment plan.

## 4. Focused visual and readability observations

- **Page 1:** the title is readable and the abstract contains the actual contribution. The contribution list continues onto page 2 in normal two-column flow; this does not obscure the thesis already stated above it.
- **Page 7:** the main GD figure and caption make the local/cumulative/pair measurements visible together. The following prose identifies their quantitative interpretation. Curves, legends and axis labels are legible in the inspected render.
- **Page 13:** the Pythia and intervention paragraphs are dense but follow a clear progression. The actual-step clipping result is present in the main discussion and immediately interpreted.
- **Page 14:** Table II is compact and readable. The diagnostic procedure, limitations and conclusion are all visible, with a clean transition to the appendix. No collision or clipping was observed on the inspected page.

The reader must still follow a mathematically substantive paper with several definitions and experimental settings. Further shortening is an editorial preference, not a discovered blocker. Another wholesale reframe would risk losing the coherence achieved here.

## 5. Enduring risks and appropriate stopping point

The scientific risks identified in round 1 remain honest qualifications:

1. The identities are elementary and the literature already contains useful oscillation, trajectory directionality and secant diagnostics. Referees may judge the integrated characterization incremental despite its clearer quantitative payoff.
2. The one-initialization full-batch sweeps support conditional demonstrations. The replicated SGD/Pythia findings do not create independent full-batch event replication.
3. The work establishes diagnostic meaning and intervention-specific responses, not a prospective improvement in decisions or a generally beneficial controller.
4. Parameter-space measurements depend on coordinate choice and do not directly measure function-space change, information retention, or generalization.

These risks should be presented to the authors as scientific judgment points. They are not reasons to commission a prediction benchmark, a larger model, or HVP calibration for claims the final manuscript does not make.

The appropriate stopping point is reached when the remaining team checks pass and the authors review this bounded contribution. Adversarial review has repaired identifiable interpretation and clarity issues. Additional simulated review rounds solely to obtain an affirmative acceptance recommendation would add no scientific evidence.
