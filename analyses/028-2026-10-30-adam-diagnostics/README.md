# Directional diagnostics under Adam

Completed October 30, 2026. Cross-run scientific synthesis and manuscript
revision; no new training, hypothesis, remote execution or cloud spending.
The user requested a practical optimizer extension after the SGD sampling
experiments, followed eventually by diagnostic-informed learning-rate decisions.
The dictated optimizer name is interpreted as **Adam**, with AdamW distinguished
where the existing language-model evidence uses decoupled decay.

The implemented section is
[Directional Diagnostics under Adam](../../manuscript/draft/sections/06-adam-diagnostics.tex).
Its claim is that the same geometric measurements answer different questions
at successive stages of Adam. Interpreting those stages together supplies
testable explanations and intervention choices; the measurements alone do not
establish stability, favorable curvature or an improved scheduler.

## Argument and structure

1. **Three views of one trajectory.** Reuse norms, consecutive cosine and
   accumulation ratios for raw gradients, first moments and adaptive directions.
   Introduce only the standard Adam recurrence and update. Explain clipping,
   bias correction, learning-rate weights and decay where they change the
   interpretation. Actual parameter displacement requires actual increments.
2. **The loss response along Adam's path.** Write the existing Taylor argument
   using the actual preceding displacement. Raw gradients still describe a
   sampled loss response, but Adam changes the direction probing that response.
   The SGD curvature interpretation of the gradient-change statistic cannot
   simply be carried over.
3. **Momentum growth, rotation and preconditioning.** Separate the incoming
   gradient's parallel and transverse components relative to the stored moment.
   Explain why increasing norm may increase momentum magnitude, rotation, or
   neither after clipping. Coordinate scaling is an additional transformation.
4. **Two empirical illustrations.** A seven-rate comparison demonstrates the
   separation between raw, moment and adaptive alignment. Matched schedule
   branches demonstrate a response to learning-rate decay and the distinction
   between alignment recovery and validation improvement. Two compact tables
   keep the comparisons explicit without importing the older figure catalog.
5. **From characterization to scheduling.** Joint diagnostic patterns motivate
   interventions. Their decision value must be tested against current rate,
   training stage, clipping and recent loss, using held-out outcomes and seeds.

Sections III--V and their equations are unchanged. Section VI follows their
experiments; Section VII concludes. The abstract, introduction and conclusion
now describe this argument. In particular, the introduction positions the
diagnostics as complementary to curvature and Edge-of-Stability work rather
than claiming that prior work cannot describe trajectories.

## Idea and evidence map

| Idea | Scientific status and appropriate claim | Consequence for the paper |
| --- | --- | --- |
| Raw gradients are already available during Adam | Exact computational fact; their values depend on the sampled batch and the path chosen by the optimizer. Monitoring consecutive or cumulative geometry still needs retained vectors and reductions. | Present practical accessibility, without claiming zero overhead or an optimizer-independent landscape measurement. |
| Raw consecutive cosine observes the loss geometry | A finite-step response combines curvature, the actual step direction, sampling and a Taylor remainder. | Use the general displacement expansion; do not relabel the SGD proxy as Adam curvature. |
| LR decay restores positive raw alignment | Analysis004 supports earlier positive smoothed alignment after faster decay in both tested sizes. The 70M recovery is temporary; its whole affected-window median remains slightly negative. | Report the conditional schedule effect, not universal positivity or a flatter attained region. |
| Raw gradient norm growth propagates into momentum | Conditional mechanism: the optimizer input, its direction relative to the previous moment, and its relative magnitude determine the response. Raw norms alone are insufficient. | Keep the mechanism, distinguish clipped input, and avoid presenting observed norm co-movement as causal identification. |
| Constructive raw alignment can accompany momentum rotation | Consecutive raw alignment does not identify alignment with the longer history stored in the previous moment. A sufficiently transverse new contribution can rotate that moment. | Propose input-versus-previous-moment alignment for a future mechanism test; it was not measured in the summarized runs. |
| Momentum can oscillate while its cosine stays positive | Exact angular interpretation. All seven runs have positive adjacent moment cosines, but late medians vary substantially. Positive sign is not guaranteed by Adam's recurrence. | Describe declining directional persistence without requiring a sign reversal or asserting a general guarantee. |
| Adaptive directions reveal preconditioning | Coordinate scaling changes direction and magnitude after temporal averaging. A difference between two cosines is not an additive causal decomposition. | Compare the streams descriptively; reserve direction interventions for causal claims. |
| Cumulative retention extends to Adam | The norm-of-sum / sum-of-norms construction applies to any recorded vector stream. Bias correction and varying LR alter temporal weights; decay adds motion. | Distinguish raw/moment/adaptive accumulation from actual parameter displacement. |
| More positive alignment means better optimization | Contradicted as a universal criterion by these examples: held-rate branches have lower final validation loss despite more negative raw alignment. Within the matched RTX 4090 cohort, the best-loss run has less aligned momentum than the next-best run. | Preserve these contrasts as part of the scientific contribution, not as inconvenient exceptions. |
| The diagnostics identify dangerous curvature regions | Not established for the LLM runs. They describe responses along the traversed path; noise and preconditioning can change the same observables. | Require a common-objective directional response or controlled local intervention before making a curvature claim. |
| Diagnostics can improve LR scheduling | A motivated, unvalidated research direction. Schedule responsiveness is not predictive or decision value. | End with an explicit evaluation requirement, not a new threshold or a claim that a controller already works. |

## How the later paper can build on this section

The next empirical step can deepen the LLM mechanism before proposing a rule.
Record the clipped optimizer input, its alignment with the preceding moment,
the adaptive direction and actual parameter increment. Keep the existing raw
diagnostics and loss. Specify parameter scope, precision, batch/noise coverage
and the training stage before drawing stream comparisons. A large batch may
reduce sampling variation, but it does not certify a noiseless measurement.

[Analysis025](../025-2026-10-17-pythia-descent-mechanism/REPORT.md) is useful
existing evidence for that deeper section: all 18 norm-matched preconditioning
ablations increased fixed-probe loss across three sizes, three seed blocks and
two branch states. It supports a role for coordinate scaling under that recipe.
Its finite-step nonlinear penalty is not a measured Hessian curvature, and it
does not establish a unique transition around updates 200--250. We retain this
as a mapped source rather than importing a second mechanism experiment into
the short Adam extension.

A later scheduling test should branch common checkpoints into predefined LR
choices, match future batches and the evaluation budget, and measure subsequent
training and held-out loss. Compare a baseline decision using recent loss,
current LR, stage and clipping with one that also uses the directional
diagnostics. Separate development checkpoints/seeds from evaluation ones;
count intervention and diagnostic costs. Establish decision improvement before
claiming an intelligent schedule. This is a proposed research direction, not
an allocated run or launch authorization.

## Evidence and reproducibility

- [Evidence observation](observations/001-adam-extension.md): source, reduction,
  coverage, captions, uncertainty and interpretation limits for both tables.
- [01_build_evidence.py](01_build_evidence.py): checks 13 original event-file
  hashes, all 4,984 sweep updates and 2,130 affected schedule pairs; regenerates
  the two manuscript tables and [evidence.json](evidence.json).
- [02_verify_manuscript.py](02_verify_manuscript.py): independently checks the
  40 displayed cosine/loss values, key numerical claims, mathematical examples,
  unchanged prior sections and compiled output; writes
  [verification.json](verification.json).
- Run both scripts from the workspace with Python, NumPy and pypdf installed.
  Event files must be restored from the retained original artifacts at the
  paths/hashes in `evidence.json`; they are intentionally excluded from Git.
  Run027 uses its original hash-verified compact retrieval copy because the
  run-folder event file is absent. No missing values are imputed.
- Build `bare_jrnl.tex` from `manuscript/draft/`, directing output to
  `manuscript/artifacts/draft/`: pdflatex, bibtex, then pdflatex twice.
  Scientific checks pass; the 17-page PDF has no overfull boxes or unresolved
  references. All pages were rendered; Section VI and the reflowed tables were
  inspected at page scale. Section VI is on pages 11--12, Tables II--III on 12.

The previous `sections/04-results.tex`, `sections/06-appendix.tex` and supplied
assets remain unchanged on disk, outside the active build. The new
`sections/06-loss-appendix.tex` retains the old loss argument without the
retired sample-level/image-result material. The full-GD appendix keeps all
scientific text, equations, captions and table contents; its three wide tables
are grouped at the end to avoid an almost-empty page and isolated final table.
The current appendices are A (alignment/loss), B (local loss derivation) and
C (full-GD measurement details). Build logs, before-snapshots and scratch page
renders remain ignored under `manuscript/artifacts/draft/adam-diagnostics-revision/`.

Primary references checked for this revision: the original
[Adam algorithm](https://arxiv.org/pdf/1412.6980),
[AdamW formulation](https://arxiv.org/html/1711.05101v3),
[Pythia paper](https://proceedings.mlr.press/v202/biderman23a.html), and
[MiniPile paper](https://arxiv.org/abs/2304.08442).
