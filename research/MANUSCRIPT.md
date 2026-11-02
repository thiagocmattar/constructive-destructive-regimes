# Manuscript and evidence

## Author-feedback revision, 2026-11-02

The [completed 28-comment revision](../manuscript/draft/revisions/2026-11-02-feedback/README.md)
reorganizes the curvature argument around directional overshoot, expands the
noise expectation, and separates the signed contributions to loss change.
The local loss-rise condition requires both sufficiently negative cosine and
gradient-norm growth. Independent two-state noise attenuation is explicitly
qualified; it is not asserted as a universal evolving-SGD law.

Section IV now pairs loss/projected change and cosine/norm/retention, with a
reformatted endpoint table, a compact Analysis035 backward profile and an
SGD batch-size subsection. The latter retains initialization and the 2k state
from Analysis027, one sampled-gradient curve per panel. Prose emphasizes
the common interpretation across objectives while retaining the observed
exceptions and keeping the detailed loss audit in Appendix C.

Four vector figures reuse existing evidence; all 52 curves / 134,456 plotted
values, nine table rows / 45 displayed scalars, and all nine backward endpoint
sign ranges pass independent checks. There are 186 deterministic equation
checks plus orthogonal/canceling-path checks. The 16-page next-review PDF is
compiled and visually reviewed, with resolved references and no overfull boxes.
The annotated original is preserved, and editable/review work now resides in
`manuscript/draft/`. No new run, training, remote execution or cost is introduced.

## Earlier manuscript records

Working title: *Constructive and Destructive Training Regimes: A Framework for
Interpreting SGD Stability*.

The original LaTeX archive has been extracted, without content changes, to
`manuscript/draft/`. Entry point: `bare_jrnl.tex`. The original archive and the
submitted PDF/reviewer materials remain in `manuscript/`. See its README and
initial source inventory.

The draft motivates questions and terminology; it does not establish results for
Pythia/MiniPile or authorize a run. Existing empirical content concerns the
original study, including image-classification training. New LLM evidence must
declare its own definitions, implementation, coverage, and limits.

The archive and review submission are separate supplied snapshots. Their exact
revision correspondence has not been established; reconcile equations/claims
against the relevant snapshot before rewriting in response to a review.

For requested edits, work in the extracted draft. Preserve original inputs,
retain the smallest useful change, and distinguish editorial correction from
changed mathematical or empirical claims. Record result-bearing changes against
observations/findings and their source artifacts. Build outputs belong in
`manuscript/artifacts/`, not beside the source figures.

The revision concerns are summarized in [REVIEW_CONTEXT.md](REVIEW_CONTEXT.md).
They are context for deciding the next work, not a prepopulated hypothesis list.

On 2026-10-04, [Analysis003](../analyses/003-2026-10-04-raw-gradient-regimes/REPORT.md)
records the user-requested assessment of a narrower central claim using completed
Pythia/MiniPile pilot evidence. Adjacent raw-gradient negativity and whole-trajectory
interaction differ, and lower loss coexists with more negative adjacent pairs at
larger size. The report proposes horizon-specific wording and discusses limits
of the lag-one and loss interpretation. No draft or supplied manuscript asset
was changed. The now-completed [three-seed audit](../analyses/003-2026-10-04-raw-gradient-regimes/observations/003-replicated-conclusions.md)
reinforces the restricted conclusion; the report retains it with updated coverage.
The first single-seed schedule intervention is now complete in
[Analysis004](../analyses/004-2026-10-04-lr-schedule-intervention/REPORT.md).
Schedule-sensitive raw-gradient cancellation coexists with lower validation
loss under held LR at both sizes. The restricted wording remains conditional
on these checkpoint continuations; extra seeds are deferred. No draft or
supplied manuscript asset was changed.

[Analysis005](../analyses/005-2026-10-04-temporal-lag-profiles/REPORT.md) addresses
item3 of the supplied reflections through calibrated all-lag measurements.
Negative adjacent raw interaction coexists with positive exact all-pair interaction
in the three middle windows; lag1 is an inadequate cumulative proxy in these
trajectories. The report recommends a claim conditional on temporal horizon and
training phase, and distinguishes the raw-vector identity from actual AdamW
displacement. The literal early/late zero-decay expectation is not uniformly
supported. One seed, numerical replay sensitivity and nonstationarity remain
explicit; the original replicated conclusions and manuscript assets are preserved.
No working TeX or supplied reflection was changed.

On 2026-10-07, the [surgical changes proposal](../manuscript/2026-10-07-surgical-changes-proposal/README.md)
records the user's preference for characterization plus a practical LR
intervention direction. It maps concrete draft changes to the existing evidence,
replaces the proposed stability framing with horizon-specific vector geometry,
and specifies the missing preconditioning-free Adam first-moment comparison.
The plan preserves the positive lag-3 rebound and the held-LR/lower-loss result,
and separates possible future feedback control from demonstrated findings.
No new experiment, registry ID, cloud execution, or draft edit is part of this
proposal; original manuscript and research evidence remain unchanged.
The user's subsequent notation feedback is incorporated: the mathematical
proposal now starts directly with mini-batch SGD gradients and displacement,
then treats Adam separately. Generic update and vector-sequence symbols have
been removed.

On 2026-10-13, the requested
[GD formulation report](../manuscript/2026-10-11-restatement/README.md) turns
the supplied formulation into an English LaTeX technical report, following the
local style guide and stopping before Displacement under Adam. It retains the
exact displacement and all-lag identities, qualifies the local curvature/noise
approximation, and adds the unbiased-noise argument and gradient-secant
interpretation. The empirical section draws on GD Runs059-062 and
[Analysis019 Observation004](../analyses/019-2026-10-13-cifar10-eos-lr/observations/004-gd-paper-figure.md).
It explicitly distinguishes path-relative cancellation from absolute displacement
and loss reduction. The report retains the proposed alignment-guided LR TODO
without claiming an executed intervention. All pages and the compact figure are
rendered and reviewed; source/evidence/output hashes are retained. Original
Markdown inputs and the existing main manuscript remain unchanged. No new
training or cloud execution occurred.

At the user's request, the report's editable TeX, generated table and builder
are colocated directly in `manuscript/2026-10-11-restatement/`. Build intermediates
remain in ignored `manuscript/artifacts/`; the delivered PDF is preserved.

On 2026-10-14, the user extended this report with a compact Adam momentum
section. It separates actual adaptive-direction displacement from stored
first-moment alignment, retains the first-order curvature/noise approximation,
and defines cumulative momentum accumulation as B/A. Bias correction preserves
adjacent cosine but generally changes cumulative accumulation. Deterministic
quadratic checks cover the momentum identity, adaptive displacement and this
distinction. No new optimizer run or Adam empirical claim is introduced.

The subsequent Rosenbrock results section reuses immutable Runs055-057, with
a publication trajectory/contour figure and a separate eight-panel diagnostic
figure recorded in Analysis017 Observation003. The momentum curvature estimate
is the Eq.25 rearrangement with its tracking term, using stored m. All 5,997
pairs, the decay/nonlinear-remainder decomposition and 22,800 complete means
are checked; the original run inputs and previous PDFs are preserved.
This is postprocessing of the existing AdamW trajectories, with no new run,
cloud execution or claim that the estimates equal maximum Hessian curvature.

The user also requested a brief noise-experiment statement and a publication
version of Analysis014's sigma comparison. It preserves all 90 original scalars
from the 15 paired conditions, with the same seed1234 and sigma0/0.01/0.1/1/10.
Observation008 records source/plot/PDF checks and single-realization limits.
No new noise draws or optimizer trajectories are generated.

On2026-10-14, the user requested real-data EOS extensions with robustness aimed
at IEEE TNNLS. The report moves noise after normalized displacement, adds
finite-population gradient variance and the fixed-parameter accumulation limit,
and includes the three-seed [Run068 control](../analyses/020-2026-10-14-eos-multitask/observations/001-frozen-real-data-noise.md).
The figure uses real sampled gradients on CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer, with32 independent128-draw sequences per condition.
This is gradient accumulation without parameter updates, not SGD trajectory
evidence. The first500-update three-case GD/SGD stage is running on owned Spark;
its empirical comparisons remain pending. The 13-page report and standalone
control figure are rendered and reviewed, with input/source/artifact hashes.
Original supplied Markdown and earlier evidence remain unchanged.

On 2026-10-17, the user requested stepwise integration of the restatement into
the working manuscript. Section 3 is now *Parameter Displacement under
Stochastic Gradient Descent*, following the restatement through normalized
displacement (Section 2.2). It develops the exact displacement/all-lag split,
consecutive cosine, directional curvature/noise approximation, finite-step
gradient projection, and B/A with explicit online measurement costs and limits.
No lag-one truncation or alignment-based stability guarantee is assumed.

The background now ends with a practical pretraining-stability review using
PaLM v5 Section 5.1 (2204.02311), OPT v4 Section 2.5 (2205.01068), OLMo 2 v3
Sections 2.3 and 3 (2501.00656), and Llama 3 v3 Sections 3.3.2 and 3.4.1
(2407.21783). PaLM supplies broader large-scale context; the other reports
document models with released weights. The reference TeX is preserved.

The exact older sample-level algebra and supplied loss/interference figure are
retained in Appendix A, and the existing two-step loss expansion is self-contained
in Appendix B. Later empirical prose and its original metrics remain for a
separate revision; only dependent references and a notation pointer are repaired.
Eleven deterministic numerical checks cover the new identities, noise/remainder
signs, reversal versus divergence, and preserved sample-level factors. The draft
is compiled and visually reviewed under ignored `manuscript/artifacts/draft/`.
These are manuscript mechanics checks, not new training evidence or experiments.

The next user-directed revision removes the alignment table and the spectral
definition of directional curvature, explaining kappa through the gradient
response projected along the preceding step. The first-order sign conditions
are stated in prose. A sampling-noise subsection now precedes normalized
displacement, reusing the existing per-sample covariance for the exact
finite-population factor and separating fixed-parameter gradient draws from
an evolving SGD trajectory. The fixed-state B/A limit follows its definitions.
No noise experiment or additional figure is inserted.

The section ends with *Connection with Loss Decrease and Convergence*, reusing
kappa in the one-step loss expansion and cosine in a compact two-step expansion.
Appendix B now uses the same g/theta/H indexing and derives these relations,
including cancellation of noise energy in the expected first-order loss term.
The local quadratic reversal boundary at eta*kappa=1 is distinguished from
the loss-descent boundary at eta*kappa=2 and from a convergence guarantee.
Fifty-five exhaustive finite-population and deterministic objective-value checks
pass with maximum absolute discrepancy 1.78e-15; the 13-page draft compiles with
resolved references and no overfull boxes. Validation scripts and records remain
under ignored `manuscript/artifacts/draft/`. These checks add no training evidence.

The author-marked opening of Section 3 is polished around gradient magnitude
and directional alignment. Displacement describes the net parameter change
from initialization; it is not equated with an amount of learned knowledge.
The slope interpretation distinguishes full gradients from sampled gradients
and first-order loss predictions from finite-step loss changes. An exact
pairwise-cosine expression replaces the separate lag-one/longer-lag presentation,
with consecutive cosine retained as its online diagnostic. Consumed editing
markers are removed. Squared-norm terminology replaces the energy wording
throughout the working draft; the separate energy-distance statistic is retained.
Six identity checks, including zero-gradient cases, pass, and the 13-page PDF
compiles with resolved references and no overfull boxes. Changed pages are
visually reviewed; the before-edit source and verification records remain in
ignored `manuscript/artifacts/draft/`. No experiment or empirical claim is added.

The author clarified that the learning intuition is essential to the motivation
for displacement. The paragraph now starts from an untrained reference and
describes the final parameters as encoding what the model learned within its
training budget. Squared displacement provides a geometric view of the magnitude
of the parameter change through which that learning is expressed; no conversion
to information units or performance guarantee is introduced. The manuscript
README records the general writing principle: introduce each variable with its
practical intuition and diagnostic purpose. The formulas are unchanged, and the
13-page draft compiles with resolved references and no overfull boxes. Reflowed
pages are visually reviewed; the previous source and PDF are retained in ignored
`manuscript/artifacts/draft/`.

The directional-curvature and noise definitions are now separate numbered
equations, each followed immediately by its own explanation. Curvature is
introduced through how the loss bends along the preceding gradient direction;
the interpretation centers on eta*kappa as the curvature effect over one
update. Short sentences explain the noiseless first-order alignment boundary
at one. Epsilon is explained through whether changing sampling noise favors
the preceding direction or opposition, shifting that boundary to 1+epsilon.
The component/projection wording in these explanations is removed. Definitions
are algebraically unchanged. The 13-page draft compiles with resolved references
and no overfull boxes; changed pages are rendered and visually reviewed.

An explicit inner-product step now follows gradient evolution: left multiplication
by the preceding gradient transpose is shown before distributing the identity
and Hessian terms. The text explains the resulting squared norm and the
normalizations leading to kappa, epsilon, and consecutive cosine. The noise
discussion points directly to the new scalar equation. The 13-page draft
compiles with resolved references and no overfull boxes; changed pages are
rendered and visually reviewed. No scientific definition or assumption changes.

The noise subsection now starts from the consecutive-cosine relation and the
distinction between the gradient-only proxy and its epsilon correction. Negative
epsilon lowers alignment and inflates the uncorrected proxy. The negative
expected noise projection concerns an unnormalized term; it does not determine
the mean of epsilon after division by a random squared norm. The exact batch-size
variance is conditioned on the current parameters. Comparable gradient norms
and weak observed alignment explain a proxy near one, without a universal
noise-to-one trend, a curvature threshold claim, or a stability guarantee.
The frozen inner-product/norm calculation is removed; the sampling control for
B/A is introduced where that ratio is discussed. Forty-two deterministic
enumeration and counterexample checks pass, and the 13-page working draft
compiles with resolved references and no overfull boxes. Changed pages are
visually reviewed. These checks add no training evidence.

The approved expectation analysis is now explicit in the noise subsection.
Taking expectations of the corrected curvature estimate and substituting the
cosine relation displays cancellation of its signed epsilon contributions.
The variance-based approximation uses squared full-gradient norm S and sampling
variance V, with the random denominator replaced by S+V as an additional
assumption. This gives an upward V/(S+V) contribution to the gradient-only proxy,
not a universal attraction of total curvature to one. The exact finite-population
variance and a scalar quadratic recurrence distinguish correcting the diagnostic
from removing fluctuations in the trajectory; the latter illustration assumes
independent additive noise with constant variance. Sixty-seven deterministic
checks pass with maximum absolute discrepancy 1.78e-15, including expectation
cancellation, normalization failure, nonlinear remainder, and scalar moments.
The 13-page working draft compiles with resolved references and no overfull
boxes; changed pages are visually reviewed. No training evidence is added.

At the author's request, the last two noise rewrites are rolled back. The
subsection now contains only the expected inner-product noise contribution and
the finite-population batch-size variance, with their short explanations.
No text outside that subsection is changed by this rollback. The 12-page working
draft compiles with resolved references and no overfull boxes; changed pages
are visually reviewed. Earlier derivations remain in the revision history.


On 2026-10-30, the user authorized an experimental section immediately after
Section III and an IEEE-oriented critical review of the Analysis024 figures.
The six isolated figures now have no embedded footers, use external panel
letters, and are sized at 7.16 inches with embedded fonts. All 54 scientific
Y arrays are unchanged; gradient indices now follow manuscript t rather than
preupdate state indices. Two composites organize the main geometric comparison.

The new Section IV, *Empirical Characterization under Full-Batch Gradient
Descent*, gives the matched nine-condition design, objective/data coverage,
three main figures and a generated endpoint table. Appendix D retains raw
cosines. The argument distinguishes reversal, path-relative accumulation and
loss reduction; it explicitly retains calibration selection, one initialization,
subset Hessians, different temporal reductions and finite-step approximation
limits. It makes no prospective instability, convergence, generalization or
measured diagnostic-overhead claim. Only obvious Section III typos and its
nonadjacent-lag inequality are corrected; its equations are unchanged.

[Analysis024 Observation006](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/006-ieee-figures-and-manuscript.md)
provides the claim-to-source map and links the individual figure review,
asset generator and verification. The 16-page working PDF compiles with
resolved citations/references and no overfull boxes; all new artwork and
reflowed pages are visually reviewed. No scientific inputs, raw evidence,
provided manuscript assets or training trajectories are changed.

On 2026-10-30, the requested second critical review of Section IV is recorded
in [Analysis024 Observation007](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/007-section-iv-critical-review.md).
A reproducible scalar audit finds positive final all-pair interaction in all
nine trajectories despite negative cumulative lag-one interaction in seven;
absolute displacement rises with LR while R falls. A correctly indexed
retrospective loss-change check exposes limits of the local approximation,
with sensitivity thresholds and interpretation limits retained. The review
prioritizes the local/all-lag distinction, theory validation, spectral scope
and a clearer contribution relative to Edge of Stability. The working draft
and publication figures are unchanged; no training or cloud work is launched.

The subsequent [Section IV surgical proposal](../manuscript/2026-10-30-section-iv-surgical-proposal/README.md)
provides exact anchors and replacement prose for eight coordinated edits.
It retains the three main figures, promotes the local/all-lag distinction,
incorporates the completed retrospective loss audit, moves spectral context
and preparation details to Appendix D, and specifies a generated displacement
column. Two small Section III edits align the geometric interpretation.
Optional additional measurements are separate from the minimum editorial
package. The proposal is local and the manuscript remains unchanged.

The user then authorized application of the proposal. All eight edits are
implemented in the working draft, with the existing three main figures,
gradient-sum displacement in Table I, and three Appendix D evidence tables.
The two Section III prose corrections preserve its equations. The retained
PyTorch vocabulary/model revision is now cited explicitly.
[Analysis024 Observation008](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md)
records the source mapping, independent verification of all 36 table rows
and 26,991 loss comparisons, preserved figure arrays and visual review of
the 18-page draft. No additional training, HVPs or cloud work was performed.

The subsequent [mini-batch noise assessment and proposal](../manuscript/2026-10-30-minibatch-noise-proposal/README.md)
reviews the compiled Section III-B and Analysis020's frozen initial-state
control. It separates the exact unnormalized noise projection and conditional
variance law from the concentrated-norm cosine approximation, and distinguishes
frozen gradient accumulation from actual displacement. The proposed MLP design
uses three initialization blocks, three trained states, five batch sizes and
paired SGD continuations with complete-objective gradient references. It includes
equation-specific figure contracts, uncertainty, checkpoint reconstruction needs,
and positioning against gradient-noise-scale and stochastic-EOS literature.
No new empirical findings, manuscript edits, training, HVPs or remote work are
claimed; no run/analysis IDs are allocated and the earlier held queues remain held.

The user approved that mini-batch design and Runpod execution because Spark was
busy. Analysis027 is now complete: three initializations, three checkpoints,
five batches,297 moving forks and 1,188 complete-objective directional HVP pairs.
[Observation001](../analyses/027-2026-10-30-minibatch-noise/observations/001-controlled-minibatch-noise.md)
provides the scientific source/reduction/caption contracts and uncertainty.

The new Section V presents frozen exact-moment calibration versus approximate
cosine, sampled and full-path geometry, paired loss/displacement contrasts and
the noisy proxy interpretation. Appendix E contains raw geometry, full loss,
noise-corrected HVP checks, coverage and qualification limits. Two Section III
prose corrections preserve all 16 equations. The historical seed 0 reconstruction
fails its predeclared tolerances; the paper explicitly uses new matched
references, whose six corresponding full-GD continuation segments replay exactly.

The [integration record](../analyses/027-2026-10-30-minibatch-noise/observations/002-manuscript-integration.md)
verifies 17 new table rows / 123 displayed scalars, 1,188 HVP pairs and 7 copied assets.
All five vector PDFs and the 23-page draft are visually reviewed, with zero
unresolved references or overfull boxes. The original assets and earlier
full-GD figure data are preserved. Existing introduction/conclusion and older
image-result alignment remain later editorial work. Task cloud resources were
deleted after local hash-verified retrieval; Spark and held queues stayed untouched.

The user subsequently rejected the first Analysis027 presentation as confusing.
The [focused editorial revision](../analyses/027-2026-10-30-minibatch-noise/observations/003-focused-manuscript-revision.md)
removes auxiliary aliases, the independent HVP argument, both dense tables and
Appendix E from the paper. Section V now asks how batch size affects alignment
at fixed parameters and how that differs during SGD. Two replacement figures
put batch size on the horizontal axis and show existing cosine/displacement/loss
quantities. Essential sampling and uncertainty methods remain in the section.
All original measurements and diagnostic figures are retained in Analysis027.
Nearby older prose is corrected where it inferred random-walk motion or
convergence from cosine/parameter norm alone, and the introduction roadmap is
updated. Section III and the older optimizer equations remain unchanged.
The 20-page PDF places both new figures on page 10 before Section VI on page 11.
All 270 initialization values, 90 plotted means, 180 range endpoints and numerical
statements pass independent checks; individual figures and page reflow are
reviewed. No new experiment or cloud execution was performed. The previous
23-page integration record is historical, not the current presentation.

The next [reviewer-facing pass](../analyses/027-2026-10-30-minibatch-noise/observations/004-temporal-interpretation.md)
condenses the frozen calibration and explicitly limits Eq. (19) to its
same-state approximation. The equations are unchanged. Section V states the
single learning rate and uses the batch-size outcomes plus a new temporal
figure from the 1,000-update continuations. It explains the positive full-GD
endpoint through transition timing and the observation window, and separates
alignment, displacement retention and objective progress. The former frozen
figure remains in Analysis027. Both current figures appear on page 10;
1,140 plotted means and 54 raw source histories pass independent checks.
The 20-page PDF is rebuilt and visually reviewed. No training or cloud work.

## Adam diagnostic extension, 2026-10-30

[Analysis028](../analyses/028-2026-10-30-adam-diagnostics/README.md) implements
the requested Section VI after the mini-batch experiments. It distinguishes
raw-gradient response, momentum growth/rotation, coordinate scaling and actual
parameter displacement, reusing the existing diagnostics without transferring
SGD curvature or stability conditions to Adam. Two compact tables summarize
the existing seven-rate Pythia comparison and matched schedule interventions.
Clipping qualifies raw-norm propagation, and more positive alignment is kept
separate from validation improvement. Scheduling is an explicitly unvalidated
next step, with an evidence/idea map for its later development.

The abstract, introduction and conclusion now follow this argument. The former
optimizer/image section and sample-level appendix remain on disk but are no
longer compiled. The retained loss appendices and full-GD details are now
Appendices A--C; the full-GD tables are regrouped without changing their content.
Sections III--V and prior figures/evidence are unchanged. All 13 source event
hashes, 40 displayed values and mathematical checks pass. The 17-page PDF is
rebuilt and visually reviewed, with Section VI on pages 11--12 and its two
tables on page 12; no overfull boxes or unresolved references. This supersedes
the older image-result alignment TODOs above. No training or remote work.
