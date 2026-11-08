# Consolidation within the author's manuscript structure

**Iteration 2 - 8 November 2026. Planning only; manuscript unchanged.**

## 1. Governing argument

The author's current paper and the inputs in `human/` define the structure and
scientific direction. This consolidation fills that structure with formulations,
evidence, figures, literature positioning and specific remaining tasks. The
first proposal gave tracing and interventions too much organizing weight. That
proposal is superseded; its evidence audits remain useful.

The paper remains **Constructive and Destructive Training Regimes: A Framework
for Interpreting SGD Stability**, with its present six sections. Constructive/
destructive behavior follows from how successive learning updates reinforce
or oppose changes in a model. Curvature, gradient magnitude, sampling and
temporal accumulation connect that behavior to stability and loss progress.
Adam, representation tracing and interventions develop this same argument.

The author's reasoning, assembled from all the supplied notes, is:

1. Training solves a learning problem by changing a model and its representations.
   Curvature describes the resulting objective and update dynamics; we also
   want to understand the learning changes expressed by those dynamics.
2. Starting from SGD's update equations, constructive/destructive regimes make
   reinforcement and reversal of successive changes explicit.
3. Their connection to directional curvature supplies a gradient-only diagnostic.
   Gradient growth and sampling explain why the same alignment sign can have
   different consequences for loss and accumulated movement.
4. GD and minibatch experiments characterize these relationships across the
   existing model/data cases. They are substantive tests of the framework.
5. The same raw-gradient computation can observe the path selected by Adam.
   The Adam section characterizes high-rate excursions, then follows the
   associated motion from outputs through hidden representations.
6. The tracing algorithm, its findings and interpretation motivate interventions.
   LR changes, clipping, beta2 changes and local displacement caps test what
   different explanations make actionable and what their tradeoffs are.

The contribution's generality comes from the relationships formulated, the
measurement capability, and insights that can be examined in other learning
problems. Formal scope and empirical coverage are stated per element. A useful
framework can connect established mathematical ingredients into an explanatory
argument; its value must be demonstrated through the questions it resolves.

**Read next:** [argument and contribution map](evidence/ARGUMENT_MAP.md) for the
gap/reference/contribution/evidence chain; [author-input map](evidence/AUTHOR_INPUTS.md)
to see how every supplied request was incorporated. The
[repository review](evidence/REPOSITORY_REVIEW.md) explains where the evidence
came from and which historical plans or presentations have been superseded.

## 2. Contribution components

These IDs identify components to consolidate, not a required eight-bullet
introduction. The current four contribution bullets remain the starting point;
the new work extends them. Each component has a positioning card in the argument map.

| ID | Intended contribution | Existing formulation/evidence | Role |
|---|---|---|---|
| C1 | Constructive/destructive characterization from learning updates | Exact SGD displacement and norm/interaction decomposition; adjacent regime definition. | The central organizing idea. |
| C2 | Curvature connection and inexpensive gradient-only proxy | Boxed alignment/curvature relation, path-average identity, error bound;017/018/024/027. | Main analytical and practical contribution, with classical secant attribution. |
| C3 | Joint reversal and amplification account of loss rises | III-E/Appendix A; quadratic relation and024 finite-step audit; Adam036/055/057; functional loss accounting062/065. | Explains oscillatory descent versus amplified excursions. |
| C4 | Sampling and batch-size effects | Noise term/variance; frozen/moving controls020/027; three-model extension037. | Core theoretical and empirical component in III/IV. |
| C5 | Controlled characterization of regime emergence and accumulation |024 GD rates and R;035 exact backward lags;056 additional all-lag checks. | Main Section IV evidence in its present order. |
| C6 | Diagnostic interpretation along adaptive trajectories | Existing Appendix A mixed-Hessian identity; current V;028/029/034/036 and Pythia evidence. | Opens V using the same diagnostic and intuitive slope-response explanation. |
| C7 | Representation-level tracing, algorithm and findings |061/062 internal investigations;063 formal workflow and cross-architecture results. | Develops the learning interpretation within V. |
| C8 | Representation-informed intervention and comparative evaluation |038-054 control development,058 comparators,059 beta2,064/065 interventions. | Completes V's practical argument, with measured benefits and tradeoffs. |

## 3. Fixed section map

Keep the existing section names and progression. M/F/T/A identifiers refer to
planning elements, not new manuscript numbering.

| Current section | Preserve | Consolidate within it | Evidence/artifacts |
|---|---|---|---|
| I. Introduction | Author-edited motivation and four current contribution components. | M01: learning/representation motivation; specific cited gaps; completed tracing/intervention results. | C1-C8; argument map; current introduction and author annotations. |
| II. Stability Analysis of SGD | Curvature/stability; minibatch stochasticity; EoS; practical large-model stability. | M02: nearest prior work where it informs these same questions. | Literature L01-L17; current II-A-D. |
| III. Constructive and Destructive Regimes | Displacement formulation and all five subsections: curvature; estimator; noise; normalized displacement; loss decrease/spike formation. | M03-M06: precise relationships, interpretations and assumptions. Keep the boxed relations, path-retention explanation and angle examples. | Current equations/Appendix A;017/018/024/027. |
| IV. Empirical Characterization of Constructive and Destructive Regimes | The author's order: convergence/curvature, destructive regime/norm/R, temporal lags, batch effects. | M08-M10: link each result to its theory; retain the four current main figures and use existing direct validation where helpful. |024/035/037;020/027/056 support. No automatic restoration of the removed endpoint table or paired C/D notation. |
| V. Raw-Gradient Diagnostics under Adam | Same raw-gradient diagnostic and its meaning along Adam's steps. | M07 and M11-M20: characterization, tracing, and interventions in the requested sequence below. |036/055/057/058;061-065; selected earlier Pythia controls. |
| VI. Conclusion | Framework, curvature/proxy, sampling/loss and controlled characterization as the main contribution. | M21-M24: what tracing and interventions add; replace outdated future-work descriptions with completed results. | Supported conclusions from this same paper. |
| Appendices/support | Path-integral derivations and measurement conventions. | Compact validation, complete populations, probe/calibration/replication details and reproducibility. | Owning analysis records; rejected dense presentations are not restored wholesale. |

### Section V: the requested development

**Opening: interpretation beyond SGD (C6).** Adam chooses the traversal direction;
the preceding raw gradient defines the projection of the gradient response.
Keep the existing diagnostic notation and the intuitive local-slope explanation.
Use the identity already in Appendix A where needed. No repeated metric
definition or new `z` notation is required.

**V-A: high-rate characterization.** Start from Analysis058 Figure 00: one high-LR
trajectory for each of MLP, ResNet and Transformer. Characterize the selected
cases with loss, eta-kappa-hat, rho and gradient growth. The requested peak table
distinguishes observations before the outgoing update, response to the incoming
update, the peak and recovery. Preserve the loss-defined event contract and
complete population when simplifying the main displays.

**V-B: instability tracing.** Follow the tracing note's reasoning: loss/gradient
oscillation -> output changes on a fixed probe -> signal correlation -> hidden
representations -> temporal SVD structure. Present Algorithm A1, layerwise
findings and interpretation as contributions. Use061/062 for the investigative
development and063 for the unified method. The heatmap/spectrum/coefficient
display follows the visual idea in065 Figure 07 page 2.

**V-C: representation-informed interventions.** Explain what each action
controls: LR scales the step; clipping changes optimizer input; beta2 changes
adaptive memory; local caps regulate a layer's response to a proposed update.
Present LR schedules, clipping, the recovered beta2 study and local caps in
that sequence. Compare each with its unchanged baseline and examine loss,
representation motion and subsequent progress. The ablation grid supports
this argument without dominating it.

These blocks implement the author's requested Adam restructuring **within V**.
Subsection lettering follows typesetting when applied; no new top-level tracing
or intervention section is planned.

## 4. Artifacts and placement

| Display | Treatment | Source / remaining work |
|---|---|---|
| Current Figures1-4 (F1a/F1b/F1c/F2) | Retain GD loss/curvature; rho/norm/R; backward lags; batch effects. |024/035/037 assets; targeted interpretation/caption checks. |
| F3 high-LR Adam overview | Use the view named by the author as the main characterization. |058 Figure 00; preserve nine-arm/population evidence in support. |
| T1/T3 timing and peak table | Eta-kappa-hat, rho, norm growth and loss with explicit availability/update columns. | P02/P04. Pre-update detection remains an explicit objective to validate. |
| A1/F4 tracing | Algorithm and output-to-layer/SVD findings, using readable stage aggregation. |063 method/figures;061/062 explanation; matched windows and probe definitions. |
| A2/F5 local cap | Local rule plus effects on loss and representation motion. |065 report/Figure10 and Figure 07 page 2; replication. Controller uses local affine scores, not the temporal SVD. |
| F6/T4 intervention comparison | LR, clipping, beta2 and caps with learning tradeoffs. |058/059/065 and older043 Transformer clipping; check fork/window matching before ranking. |
| Supporting validation | Compact evidence for the analytical claims. |027 HVP/noise checks;024 loss audit;056 all-lag evidence. Select presentation with the author; no automatic restoration of dense tables. |

There is no new fixed figure/page quota overriding this structure. Balance space
by consolidating existing Adam displays and using supporting material deliberately.
[ARTIFACTS.md](evidence/ARTIFACTS.md) provides paths and generators.

## 5. Existing support and actual remaining work

- **Curvature/noise:**027 has 1,188 state-matched HVP pairs. At batch 32 the raw
  proxy median absolute discrepancy .7467 becomes .0030 after measuring the
  explicit noise correction. This independently computed derivative comparison
  supports the decomposition; the correction itself is an offline reference.
- **Regime accumulation:**024/056 show negative adjacent interactions alongside
  positive all-pair interaction and continued progress. This develops the
  interpretation of local destructive behavior and R.
- **Batch effects:**037 already has the three-model, three-initialization
  extension.020/027 isolate fixed-state sampling and compare moving paths.
- **Adam/representations:**036/055/057 supply characterization and timing;
 061/062 connect amplification to functional/hidden motion;063 supplies the
  tracing method.064/065 include freezing, selective LR, norm/directional
  controls, cross-architecture caps and three-seed MLP ablations.
- **Comparators:**058 consolidates schedules/clipping;059 is the beta2 study.
 043 also contains short Transformer clipping tests at a different fork.
  Missing is matched clipping in the selected tracing/cap window.

The immediate work is P01-P05: finish source-to-claim and prior-work comparisons,
select concise validation, build the specified peak table, and prepare the
requested tracing/comparison figures. [TASKS.md](work/TASKS.md) provides assignments.

New measurements must answer a named missing link: matched controls at the
chosen cap forks, missing larger-model replication, probe robustness, measured
cost for efficiency claims, or a time-correct test of the requested pre-update
warning objective. A comprehensive new optimizer benchmark is not imposed on
the paper. Completed work and stale TODOs are separated in the repository review.

Qualifications stay local to the relevant claim: frozen sampling versus moving
SGD, path-average versus spectral curvature, training versus held-out outcomes,
and retrospective observation versus prediction. They make the author's chosen
argument precise rather than determine a different emphasis.

## 6. Review procedure and next deliverable

Use the supplied review-skill to check each chain:
**question -> cited predecessor -> specific boundary -> contribution -> direct
evidence -> significance -> scope/remaining test**. The argument map gives the
affirmative positioning; [CLAIMS.md](evidence/CLAIMS.md) checks mathematical and
empirical support. Neither document selects a different paper structure.

The next author review concerns mapping completeness, proposed content within
each section, positioning and specific remaining tasks. Structure and central
emphasis are no longer open choices. Continue consolidation until the author
is satisfied; apply manuscript changes only when requested. No training,
cloud spending or publication is authorized by this document.

A separate submission item is recorded: the supplied previous TNNLS decision
restricts revised resubmission. W02 must resolve the intended submission route
against that letter before submission. This does not change the author's venue
goal or interrupt scientific consolidation; no alternate venue is selected.
