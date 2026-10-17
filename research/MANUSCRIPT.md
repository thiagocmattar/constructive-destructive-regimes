# Manuscript and evidence

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
