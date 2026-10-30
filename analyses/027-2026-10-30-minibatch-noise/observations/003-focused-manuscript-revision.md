# Observation003: critical review and focused manuscript revision

Date: 2026-10-30. Editorial revision requested by the user after reading the
first integration. No training, cloud work or scientific-input changes.
This record supersedes Observation002's manuscript presentation, while retaining
its measurements, checks and original diagnostic figures as research evidence.

## Critical assessment

The first integration passed arithmetic and layout checks but did not present
a clear scientific argument. Four problems mattered to a reviewer:

1. The section introduced `S`, `V_b`, `q`, `mu`, `chi_b`, and new time indices
   even though Section III already defined the quantities needed to interpret
   the experiment. Readers had to translate notation before assessing evidence.
2. Predicted-versus-observed normalized moment plots obscured the intervention:
   batch size. Dense time curves then mixed sampled gradients, full gradients,
   displacement and a second curvature-validation question.
3. The HVP correction study, exact accounting residuals and reconstruction
   audit occupied the paper without a proportional role in the sampling argument.
   The appendix repeated plots and introduced an additional identity. It read
   like an experiment log rather than supporting a specific claim.
4. The new SGD figure floated into the next section, whose older prose also
   equated near-zero cosine with random-walk motion. The presentation and the
   nearby interpretation contradicted the distinctions the experiment needed.

The revised argument has two steps: hold parameters fixed to isolate sampling,
then resume SGD to measure its consequences for alignment, displacement and
loss. The first comparison tests Section III-B. The second explains why the
same cosine cannot automatically be read as curvature or objective progress.

## Applied changes

Section V is now *Effect of Mini-Batch Sampling on the Diagnostics*. It uses
the existing notation, gives the experimental design and uncertainty directly,
and explains each comparison before interpreting its result. Auxiliary aliases,
the HVP discussion, two dense tables and the added Appendix E are removed from
the compiled paper. The independent HVP measurements and all original figures
remain in Analysis027; their scientific results have not been withdrawn.

The two replacement figures have batch size on the horizontal axis. All five
batch sizes, three initializations and three starting states are retained.
Both figures appear on page 10, before Section VI starts on page 11. Section V
starts on page 9. The introduction's section roadmap is corrected. Three nearby
interpretive paragraphs in the older SGD section now describe alignment without
asserting random-walk behavior, overshooting or convergence from cosine/norm alone.
The older experiment's equations and numerical figures are unchanged.

## Figure: batch size at fixed parameters

Artifact: [batch_size_fixed_parameters.pdf](../figures/paper/batch_size_fixed_parameters.pdf).
Generator: [09_paper_figures.py](../09_paper_figures.py).
Sources: unchanged verified `results.json` and `reduction.json`, specifically
the frozen sequence means, full-gradient squared norms and complete-population
covariance predictions. The new manifest records their SHA-256 values.

Reduction: average the 32 sequence means within each initialization/state/batch;
then average the three initialization means equally. Evaluate Eq. (19) separately
at each state before averaging predictions. Blue circles show observations,
open orange squares show predictions, and shading shows the observed range
across the three initialization means. Each sequence contains 128 draws with
127 defined adjacent cosines. All 45 saved-state/batch conditions are present.

Caption meaning: batch size changes consecutive-gradient cosine even when
parameters do not move. At full batch every draw is the same gradient.
Checkpoint columns expose the dependence on the training state. Batch axes are
logarithmic and connecting lines guide the eye between measured values.

Uncertainty and caveats: shaded ranges are descriptive, not confidence intervals.
Whole-sequence conditional bootstrap intervals are used for the numerical
approximation assessment in the text. Nearly overlapping prediction markers
do not establish exact agreement: maximum error 0.0075 and 21/36 stochastic
intervals excluding the approximation are retained. The separate exact variance
and noise-projection checks remain in the prose and research evidence; no new
normalization or fitted constant is introduced for the plot.

## Figure: batch size during SGD

Artifact: [batch_size_sgd_outcomes.pdf](../figures/paper/batch_size_sgd_outcomes.pdf).
Generator and sources: the same script and verified result/reduction files,
using all 297 moving continuations and 59,400 updates.

Reduction: each continuation contributes its mean over 199 adjacent cosines,
endpoint `R_200`, and final-minus-initial full-objective loss. Sampling repetitions
are averaged within initialization (eight stochastic, one deterministic full-GD),
then the three initialization means are averaged equally. The top row also
applies the same cosine formula to full gradients at the same SGD states.
Orange is this measurement reference, not a separate GD trajectory; the
full-batch continuation is the point at batch 5,000. The middle and bottom rows
show the SGD outcome itself. No sliding temporal smoothing is applied.

Caption meaning: compare alignment, retained displacement and loss progress
under the same batch-size intervention. The initial-state comparison separates
negative sampled alignment from positive average full-gradient alignment.
The later-state comparison shows weaker reversal without a larger retained
displacement fraction, and includes a loss increase at batch 32. Batch 2,048
also outperforms full batch on endpoint loss in that later comparison, preventing
a universal monotonic-performance reading.

Uncertainty and caveats: shading shows the range of three initialization means,
not independent-pair uncertainty. Common checkpoint indices do not identify
common dynamical regimes. Time averages do not imply stationary behavior or
describe every instantaneous update. The loss panels use different vertical
ranges, explicitly stated in the caption; all measured seed means and ranges
are inside their axes. Equal updates entail unequal data exposure.

## Numerical and visual review

[10_verify_paper_revision.py](../10_verify_paper_revision.py) independently
reconstructs all 270 initialization-level values from repetition-level results,
all 90 plotted means and 180 range endpoints. It verifies the 11 selected
quantitative contrasts, moment-error bounds, cosine approximation error,
21/36 interval count and all three paired loss comparisons. The complete
Section III file is byte-identical to its pre-revision snapshot. The older
optimizer equations are unchanged. No executed source, configuration or raw
evidence is edited.

The two PDFs use 7.16-inch width, embedded Times/STIX fonts, solid lines,
external panel letters and no footnotes. They were individually rendered at
150 dpi and inspected again at manuscript size. Two-line Y labels avoid overlap
in the compact SGD grid. An axis-bound assertion caught and corrected clipping
of the initial-state loss range. Caption shortening and a small fixed-parameter
figure-height adjustment make both floats fit without shrinking fonts.

The 20-page draft compiles with zero overfull boxes and unresolved references.
Affected pages were inspected directly and remaining pages reviewed for reflow.
[paper_revision_verification.json](../paper_revision_verification.json) and
[paper_revision_review.json](../paper_revision_review.json) record hashes and
checks. Build logs, renders and the prior draft snapshot remain under ignored
`manuscript/artifacts/draft/noise-editorial-revision/`.

The scope is the sampling section and its immediate consistency problems. This
revision does not constitute a fresh validation of all legacy optimizer claims
or a claim that the whole paper is submission-ready.
