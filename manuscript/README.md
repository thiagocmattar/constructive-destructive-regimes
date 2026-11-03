# Manuscript workspace

## Active research program

The [temporal geometry and intervention program](2026-11-03-research-program/README.md)
expands the closing plan into new observability, matched-intervention and
independent-confirmation experiments. Its [execution specifications](2026-11-03-research-program/EXECUTION.md)
and [first local pilot design](2026-11-03-research-program/pilot_plan.json) record
scientific comparisons, budget gates and remaining launch preparation.

## Paper evidence and submission plan

The [3 November consolidation](2026-11-03-paper-convergence/README.md) maps the
current argument to evidence, advisor discussion and simulated reviews. Its
[bounded work queue](2026-11-03-paper-convergence/NEXT_ACTIONS.md) distinguishes
existing-data checks, optional interventions and the submission stopping rule.
This planning record does not change the current draft or authorize training.

Latest revision (2026-11-03, original argument restored):
[review and reproduction](draft/revisions/2026-11-03-original-argument-restored/README.md).
The title, abstract, keywords and main text return to `bba6203`, before the
full writing revision. That snapshot already includes the updated methods,
controlled GD/SGD experiments and complete Adam spike analysis. The introduction
starts with "Neural scaling laws have ushered in new paradigms for interpreting
deep learning dynamics." The shortened appendices and current figures/tables
are unchanged; broken references are intentionally retained at the author's
request. The separate 15-page PDF is verified and visually reviewed.

Superseded baseline restoration (2026-11-03, Sections I-IV):
[review and reproduction](draft/revisions/2026-11-03-sections-i-iv-restored/README.md).
Sections I-IV exactly match commit `a4bbf0d`, immediately before the
curvature-focused pass. The abstract, Adam section, conclusion, concise
appendices and scientific assets retain their current contents. The separate
14-page PDF passes source, evidence, compilation and visual checks; the previous
revision and its PDF remain unchanged.

Previous revision (2026-11-03, full-paper writing):
[review and reproduction](draft/revisions/2026-11-03-regime-writing-revision/README.md).
The current 14-page manuscript implements the author's writing principles and
all 12 annotated comments. It develops the constructive/destructive-regime
argument consistently from the abstract through GD/SGD and Adam spike anatomy,
strengthens contribution positioning, and improves the grouped anatomy table.
The initial pass reviewed 135 prose units and 14 captions. The subsequent appendix
review consolidates the loss derivation, condenses measurement conventions, and
removes three supplementary tables and one repeated figure from compilation.
The appendices contain no figures or tables. Experimental assets are preserved;
eight main figures and two tables remain compiled. The latest pass restores
curvature estimation without Hessian computation as a central contribution,
with a formal path-average identity, an error bound and one-gradient online
storage. It reconnects the diagnostics to the original motivation of explaining
stabilization heuristics. Checks and task lists are in the revision folder.

Previous revision (2026-11-03, Adam spike anatomy):
[review PDF](draft/revisions/2026-11-03-adam-spike-anatomy/revised-manuscript.pdf),
[scope and reproduction](draft/revisions/2026-11-03-adam-spike-anatomy/README.md).
The Adam section now follows the approved loss overview, largest-spike examples,
fixed-reference event definition, all-amplitude density, grouped >10% anatomy
table and cosine/norm-product regressions. Four Analysis036 PDFs are copied
without modification. The abstract, introduction, conclusion and Adam appendix
follow the descriptive evidence; the prior forecasting exposition is no longer
compiled. The review has 18 pages and nine active figures. Source data, earlier
reviews and the GD/SGD sections/figures are preserved.

Earlier author-feedback revision (2026-11-02, Adam section):
[review PDF](draft/revisions/2026-11-02-adam-feedback/revised-manuscript.pdf),
[TODO](draft/revisions/2026-11-02-adam-feedback/TODO.md), and
[comment dispositions / reproduction](draft/revisions/2026-11-02-adam-feedback/README.md).
Six new annotations replace the old Adam exposition/Pythia illustrations with
Analysis036: raw-gradient information about impending loss excursions beyond
recent loss, despite Adam's transformation of those gradients. Two compact
figures and a forward-test table use existing verified evidence. The experimental
scope remains in the methods; the SGD batch-size figure is unchanged.

Earlier author-feedback revision (2026-11-02):
[review PDF](draft/revisions/2026-11-02-feedback/revised-manuscript.pdf),
[completed TODO](draft/revisions/2026-11-02-feedback/TODO.md), and
[comment dispositions / reproduction](draft/revisions/2026-11-02-feedback/README.md).
The 28-comment source, revision scripts and review PDF are now under `draft/`.
The latest 16-page version includes the completed Analysis037 three-case SGD
figure, initialization/2k curves, GD layout fixes, backward-lag zoom and Table I
multirow grouping. All 348 new trajectories and the manuscript checks pass.
This revision builds into ignored `tmp/pdfs/` and delivers to `draft/`,
superseding the older `artifacts/` output convention for the current review.

The 2026-11-02 build used seven figures, including two Analysis036 manuscript views;
the current build uses eight, including the four approved Adam figures.
The extra raw GD cosine figure remains available but is no longer compiled.
The other 26 are preserved in the [figure archive](archive/2026-11-02-unused-figures/README.md),
with hashes and original paths. Inactive legacy sections reference that archive.

The original supplied materials are preserved:

- `latex-zip.zip`: original LaTeX source archive.
- `2026-10-03 review/`: submitted paper PDF, reviewer PDFs, and editor feedback.
- `source_inventory.json`: initial archive hash and per-file byte sizes/SHA-256.

`draft/` is the editable copy extracted from the archive on 2026-10-03. All 22
files initially match the archive bytes. The inventory is an immutable baseline,
not a manifest to refresh after editorial edits.

Entry point: [draft/bare_jrnl.tex](draft/bare_jrnl.tex). Section text is in
`draft/sections/`; the concise curvature/loss derivation is in `06-loss-appendix.tex`. Existing figure
PDFs and `bibliography.bib` are carried over exactly. Keep original assets until
their replacement is explicitly part of an editing task.

When introducing a variable, explain its practical intuition and why it is
useful before or alongside its formal definition. The intended audience uses
the theory to interpret training behavior. Preserve the author's motivation
when polishing, while distinguishing an intuitive interpretation from a proved
quantitative claim.

The archive and submitted PDF are distinct snapshots until their correspondence
is checked. Read the applicable snapshot when addressing a reviewer equation.
The review map is in [research/REVIEW_CONTEXT.md](../research/REVIEW_CONTEXT.md).

Build from `draft/` so relative inputs resolve; direct output into ignored
`manuscript/artifacts/`. The IEEEtran class/bibliography styles and TeX packages
must be available in the selected TeX environment. No manuscript build was run
during bootstrap. New empirical text/figures must trace to this workspace's
run/analysis observations; see [research/MANUSCRIPT.md](../research/MANUSCRIPT.md).

## Revision proposal

[Surgical changes proposal, 2026-10-07](2026-10-07-surgical-changes-proposal/README.md)
organizes the advisor discussion around temporal interference and learning-rate
interventions. It contains an evidence map, section-level changes, mathematical
definitions, an experiment plan, and a focused literature comparison. It is a
planning record; the draft, supplied reflections, and original assets are unchanged.

## GD formulation report

The [restatement report](2026-10-11-restatement/README.md) develops the supplied
formulation in English through the empirical full-batch GD result, stopping
before the optimizer extension. It follows the local writing-style guide and
includes a compact publication view of Analysis019 GD evidence. Editable TeX,
the builder, reviewed PDF and provenance are together in
`2026-10-11-restatement/`. Its source remains a separate reference for the draft revision.

The restatement is now being integrated into `draft/` in user-directed steps.
The background includes a practical pretraining-stability review, and Section 3
uses displacement, directional curvature, sampling noise, normalized
displacement, and a local connection to loss decrease. The older
sample-level algebra is retained in Appendix A for the existing image results.
Appendix B derives the loss relations using the main section's notation.
The scientific interpretation of the image results, the introduction, and the
conclusion still require later alignment with the new formulation. Current
compiled output and mechanical verification records are in ignored
`manuscript/artifacts/draft/`.

The new Section IV presents the completed Analysis024 full-batch learning-rate
comparison: matched design, three main figures, all nine endpoint rows and
explicit interpretation limits. Appendix D retains unsmoothed alignment.
The [individual IEEE figure review](../analyses/024-2026-10-17-eos-gd-lr-calibration/FIGURE_REVIEW.md)
and [integration observation](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/006-ieee-figures-and-manuscript.md)
record its scientific and visual review. Analysis024's asset generator copies
the reviewed vector figures and regenerates `draft/tables/analysis024_endpoints.tex`
from verified results; the working draft remains self-contained for building.

The [Section IV surgical proposal, 2026-10-30](2026-10-30-section-iv-surgical-proposal/README.md)
turns the second critical review into eight anchored edits with replacement
wording, a one-column endpoint-table change, appendix evidence contracts and
two Section III terminology corrections. It is now applied to the working draft.
The [implementation record](../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/008-applied-section-iv-revision.md)
documents the four generated tables and independent checks of all 36 rows and
26,991 loss comparisons. The current 18-page PDF and its affected pages have
been rendered and reviewed; the three main figures are unchanged.

The [mini-batch noise proposal, 2026-10-30](2026-10-30-minibatch-noise-proposal/README.md)
assesses Section III-B against the existing initialization-only sampling control
and proposes frozen trained-state measurements plus matched SGD continuations.
It specifies an MLP comparison, full-gradient noise references, equation checks,
two figure designs, uncertainty and a focused stochastic-EOS literature comparison.
The user approved execution on Runpod because Spark was busy. The completed
[Analysis027 report](../analyses/027-2026-10-30-minibatch-noise/REPORT.md) records
the measurements and original diagnostic figures. After the user's critical
review, the [focused revision](../analyses/027-2026-10-30-minibatch-noise/observations/003-focused-manuscript-revision.md)
rewrites Section V around two comparisons: batch size at fixed parameters,
then batch size during SGD. Two new figures show the intervention directly
using existing cosine, displacement and loss quantities. The added Appendix E,
HVP discussion and dense tables are removed from the paper; essential methods
are in the main section and all original evidence remains in the analysis.
The adjacent older SGD interpretation and introduction roadmap are corrected.
Section III and the optimizer equations are unchanged. The 20-page PDF has
both new figures on page 10 before Section VI, with zero overfull boxes or
unresolved references; all 90 plotted means and 270 initialization values pass
independent checks. No new training or remote work was performed.

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
