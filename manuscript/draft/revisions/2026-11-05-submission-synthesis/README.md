# Submission synthesis and manuscript revision

5 November 2026. This revision closes the current diagnostic argument using existing evidence. It preserves the constructive/destructive scientific question while distinguishing adjacent cancellation, cumulative motion, optimizer transformations and objective progress. No training, HVP campaign, cloud execution, or new optimizer was commissioned.

## Read in this order

1. [Scientific synthesis](SYNTHESIS.md): definitions, formulations, experiment families, supported findings and limits.
2. [Central argument](CENTRAL_ARGUMENT.md): motivation, credited literature gap, contributions, evidence hierarchy and original-rejection response.
3. [Revised manuscript](revised-manuscript.pdf): review copy of the active `manuscript/draft/bare_jrnl.pdf`.
4. [Review resolution](REVIEW_RESOLUTION.md): adversarial review findings, repairs and remaining risks.

## What changed

- Retained the constructive/destructive framework and the GD/SGD evidence, while replacing the title's general stability promise with diagnostic scope.
- Organized the theory around all-lag displacement and paired cancellation before its curvature interpretation. Credited the exact connection to the classical BB1 secant and restored closely related gradient-diagnostic literature.
- Rewrote the introduction and conclusion around one question: where motion cancels, what residual accumulates, and what the loss establishes separately.
- Condensed Adam event analysis around raw-gradient versus applied-update semantics. Kept the negative severity-transfer finding, moved the detailed memory/HVP comparison to an appendix, and removed four figures (the definition sketch, severity density and two regression displays) from compilation. Their original assets remain unchanged.
- Added a compact synthesis of existing Pythia schedule/scaling studies and matched learning-rate/clipping interventions, with explicitly different objectives, seeds and horizons.
- Preserved supplied previous-submission PDFs, prior revisions, figures, tables and raw experimental records. The initial source hashes and Git baseline are in [baseline_manifest.json](baseline_manifest.json).

The earlier [Adam experiment plan](../2026-11-05-human-input-adam/EXPERIMENT_PLAN.md) remains a proposed future study. Its HVP calibration, independent-seed mechanism study and larger-scale monitoring comparisons are not prerequisites for the bounded claims made here. It is not authorization for execution.

## Evidence and review

The [evidence inventory](reviews/evidence-inventory.md) and [selected source manifest](evidence_selection.json) provide claim provenance. [Theory/novelty](reviews/theory-novelty.md) and [advisor](reviews/argument-advisor.md) assessments informed the framing. Later review rounds assess the revised text rather than merely repeat those recommendations. These are simulated internal reviews, not actual journal decisions or acceptance probabilities.

No new run, analysis or hypothesis ID is allocated: the task revises and consolidates the manuscript without creating a new scientific experiment or cross-run fitted analysis. Existing research registries and unrelated local work are preserved.

## Reproduce checks and PDF

From repository root:

```powershell
python analyses/056-2026-11-05-gd-paired-displacement/02_verify.py
python analyses/057-2026-11-05-adam-response-memory/02_verify.py
python manuscript/draft/revisions/2026-11-05-submission-synthesis/verify_implications.py
python manuscript/draft/revisions/2026-11-05-submission-synthesis/01_build_and_verify.py
```

The original Pythia schedule is implemented in [src/baseline/recipe.py](../../../../src/baseline/recipe.py): `recipe_learning_rate` uses the pre-step clock `k=t-1`, a warmup length `w=7.12`, `b=min(k,712-w)`, linear warmup `0.001*b/w`, then `max(0.0001,0.0005*(1+cos(pi*(b-w)/712)))`. The minimum clamp may activate before update 712; the 350-update native parents retain this original clock. The complete source/config hashes remain in the owning runs and the selection manifest.

The build uses the installed MiKTeX and Poppler, writes temporary output to ignored `tmp/pdfs/submission-synthesis-2026-11-05`, checks new mathematical relations, verifies protected assets/source hashes, and renders every page. Numerical/source checks cannot replace visual inspection. The final build and visual-review records are `verification.json` and `visual_review.json`.

This is a reviewed scientific draft for author assessment. Venue-specific packaging, final author/coauthor approval and submission remain author decisions. The present task does not push, publish or submit the paper. The remaining novelty/significance and replication limits are stated in the review resolution rather than concealed by simulated favorable verdicts.
