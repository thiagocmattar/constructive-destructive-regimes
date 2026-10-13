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
