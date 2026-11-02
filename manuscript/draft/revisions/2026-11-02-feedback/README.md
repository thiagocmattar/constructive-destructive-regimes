# Next manuscript version: author feedback of 2026-11-02

The three-case SGD extension is complete in
[Analysis037](../../../../analyses/037-2026-11-02-sgd-batch-extension/REPORT.md).
All 348 new continuations and 198 reused MLP continuations supply the requested
one-row mean-cosine figure. The 16-page manuscript passes numerical and visual review.
All 13 task Pods/disks are deleted; cost US$26.30 provisional, within the approved
US$40/six-hour envelope. Spark's isolated 1.379 GB project is retained on owned
storage with no active task measurement process.

Review [revised-manuscript.pdf](revised-manuscript.pdf). The editable entry point
is [../../bare_jrnl.tex](../../bare_jrnl.tex); the usual compiled copy is also
available at `manuscript/draft/bare_jrnl.pdf`. The source comments and revision
work now live beside the draft, outside `manuscript/artifacts`.

[TODO.md](TODO.md) records the task; [comments.json](comments.json) preserves all
28 annotations and their original page/selection metadata. The unchanged
annotated input is [source-comments.pdf](source-comments.pdf). C27 and C28 are
overlapping comments, both addressed by the same SGD revision.

## Comment dispositions

**Figure-layout follow-up:** the two GD composites now place panel letters
with the column titles and the learning-rate legends directly beneath, with
compact spacing. Training-loss ticks use decimal notation on the unchanged
log scales. The backward-lag figure adds a second row zooming into lags 1--32;
all 96 zoomed values repeat the full-range data exactly. The caption and checks
describe both rows. [The archive](../../../archive/2026-11-02-unused-figures/README.md)
preserves 26 unused figure copies and their hashes, leaving only the five
compiled assets in the active figure directory. The layout-only changes
reuse existing GD evidence.

**Subsequent feedback:** the SGD figure now has one row of three mean-cosine panels, with initialization and 2k-state curves per panel. The
[extension contract](SGD_EXTENSION.md) and Analysis037 record the new ResNet/
Transformer measurements, runtime qualification, full-cohort acceptance and
cleanup. The superseded MLP-only draft copy is preserved in the figure archive.

| Comments | Revision | Scientific handling |
| --- | --- | --- |
| C01 | Section III is now *Characterizing Constructive and Destructive Regimes*. | Names the purpose of the formulation. |
| C02--C03 | Removed the two marked displacement clauses. | Preserved the net-motion definition. |
| C04--C06 | Curvature/projection interpretation precedes the diagnostic; added the local directional minimum at step size 1/kappa. | Uses the existing preupdate indexing. Orthogonality requires a nonzero endpoint gradient; zero gradients have undefined cosine. |
| C07--C08 | Expanded the preceding noisy gradient inline and removed the normalized-expectation aside. | Retained fresh conditional unbiasedness and finite second moments. |
| C09--C10 | Replaced the verbose discussion with independent two-state attenuation and its same-state special case. | Either sign attenuates under the stated concentration approximation. An evolving SGD path is coupled to its sampled updates; the same-state equation is not a negative-alignment law. |
| C11 | Explained R near zero/one, with orthogonal and opposing-step examples. | Low retention is expected conditionally on sustained reversal/incoherence; it does not imply small absolute displacement. |
| C12--C13 | Rewrote the loss change using delta-L and three underbraced contributions. | Used “directional interference”: the term can reinforce or oppose descent. For negative cosine its contribution to loss change is positive. |
| C14--C15 | Added the joint norm/cosine loss-rise inequality and the threshold at two. | Both negative cosine and norm growth are necessary within the noise-free quadratic prediction; neither alone is sufficient. Negligible noise is an assumption, not inferred from a nominal batch size. |
| C16--C19, C22 | Condensed the empirical goal/design and replaced enumerated values with interpretation. | Preserved matched initialization, training-selected rates, objectives and coverage in one main paragraph plus the appendix. |
| C20 | First figure is loss/projected change in a 2-by-3 grid. | The diagnostic requires no Hessian evaluation but concerns realized directional change, not the largest eigenvalue. The plot is descriptive, with no validated exact-onset prediction. |
| C21 | Second figure is cosine/norm/retention in a 3-by-3 grid; endpoint discussion follows. | Reversals and inefficient path retention coexist with greater absolute displacement and lower final loss at the largest rate. The evidence does not support claiming slower GD from retention alone. |
| C23 | Added a 2-by-3 backward-lag figure with a 1--32 zoom and a short interpretation. | Common middle endpoint in all models; all 256 pairs retained. State 2256 corresponds to gradient index 2257. Separate numerical continuations are identified explicitly; all nine endpoints support the appendix statement. |
| C24 | Added dataset, T notation, explicit late-window definition and decimal formatting to Table I. | A star denotes a positive value below 0.00005 rounded to zero; full-precision values remain in the source. |
| C25 | Moved the detailed loss-surrogate audit entirely into Appendix C. | Its equation, counts and limitation remain available without interrupting the main argument. |
| C26--C28 | SGD is a short subsection with the compact one-row three-case cosine figure, two starting-state curves per panel and explicit full-GD endpoints. | Smallest batches weaken trained-state reversal. The Transformer example shows that similar cosine can accompany opposite training-loss changes. The covariance/implicit-bias connection remains qualified; generalization was not measured. |

The small accompanying changes repair the introduction roadmap, add a
conclusion label, and make the background consistent with the distinction
between oscillation and instability. The classical strict-contraction
equivalence now explicitly assumes a positive-definite Hessian.

## Evidence and reproduction

Analysis024 GD arrays and Analysis035 backward profiles supply the retained GD
views. Runs214/216 add 348 ResNet/Transformer continuations and six fresh GD
references, while Analysis027 supplies the unchanged MLP continuations.
All 57 curves / 134,560 plotted points, the displayed scientific example,
runtime-bridge value, 45 table scalars and 186 equation checks pass.
Local figure observations are:

- [GD composites and endpoint table](../../../../analyses/024-2026-10-17-eos-gd-lr-calibration/observations/009-author-feedback-figures.md).
- [Three-case SGD figure](../../../../analyses/037-2026-11-02-sgd-batch-extension/observations/001-three-case-sgd.md); [earlier MLP-only view](../../../../analyses/027-2026-10-30-minibatch-noise/observations/005-compact-author-feedback.md) remains historical.
- [Backward-lag manuscript view](../../../../analyses/035-2026-11-02-eos-backward-anchors/observations/002-manuscript-view.md).

From the repository root:

```text
python manuscript/draft/revisions/2026-11-02-feedback/01_prepare_figures.py
python manuscript/draft/revisions/2026-11-02-feedback/02_build_and_verify.py
```

The first script regenerates the four figures, manuscript copies and endpoint
table, leaving the earlier artwork intact. It records source and output hashes
in [figure_manifest.json](figure_manifest.json). The second independently
checks the rearranged curves against original GD event files, SGD repetitions
and the backward CSV; checks table rounding and revised mathematics; compiles
with pdflatex/BibTeX; and renders every page and figure under ignored
`tmp/pdfs/manuscript-feedback-2026-11-02/`. It resets visual-review status on
rebuild. [verification.json](verification.json) and [review.json](review.json)
identify the delivered version and completed visual review.

The literature connection was checked against the primary sources:
[Cohen et al., EoS](https://arxiv.org/abs/2103.00065) and
[HaoChen et al., noise covariance and implicit bias](https://proceedings.mlr.press/v134/haochen21a.html).
The latter result has a specific model/noise setting; it motivates the question
of bias, not a generalization claim for the present continuation experiments.

The earlier analysis-specific manuscript validators describe older snapshots.
Use the revision scripts above for this manuscript; raw-evidence verification
and historical records remain in their owning analyses.

## Table I grouping follow-up

Table I uses `\multirow` for the shared dataset and model labels. The MLP and
ResNet-32 blocks share one CIFAR-10 cell; a partial rule separates their model
groups. All 45 numerical entries remain unchanged and pass source verification.
The grouping and all 45 values remain verified in the latest manuscript;
the table and its surrounding layout were reviewed.

Latest layout follow-up: show only mean cosine in the SGD figure. All six
retained curves and their ranges are unchanged; the 84 block scalars and 28
plotted means pass independent checks. The compact caption and main prose now
focus on adjacent alignment and training-loss outcomes. Retention data remain
in Analysis037. No additional training or cloud execution.
