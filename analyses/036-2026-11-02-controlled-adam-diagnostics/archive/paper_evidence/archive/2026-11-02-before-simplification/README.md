# Controlled Adam: evidence and proposed paper argument

Prepared for author review before another manuscript edit. All evidence reuses
the nine completed Run 215 trajectories. There is no new optimizer training,
cloud spending, Hessian calculation, initialization sweep or intervention.

The proposed contribution is **raw-gradient geometry remains informative about
subsequent loss excursions along an adaptive trajectory**. The contribution
has two empirical parts: a recurring temporal pattern and incremental forward
prediction under explicit loss-history controls. A mechanism and a useful
intervention remain open questions.

## Recommended narrative and allocation

| Role | Question answered | Main evidence | Scope of the inference |
| --- | --- | --- | --- |
| Context | What trajectories and rates are being compared? | Compact overview of all nine arms | Describes training, including smooth portions and transient excursions |
| Micro view | What changes, and in what order, near a loss excursion? | Three unsmoothed examples, with raw cosine and both norms | Conspicuous, explicitly loss-selected cases; teaches the reader how to read the diagnostics |
| Systematic timing | Does the sequence recur? How often is it absent? | All-arm episode census, lead times, false-alarm audit; onset-aligned repeated events | Conditional recurrence and specificity in these trajectories |
| Incremental prediction | Is the diagnostic informative after observing recent loss? | Identical forward-test states, nested feature comparisons, temporal blocks, richer-loss and delay checks | Incremental information under the tested forecasting models; does not identify a cause |

Items 2 and 3 in the discussion are complementary. The temporal census estimates
how often an event has a preceding signal. Forecasting asks whether the signal
improves predictions beyond information already in loss, task/rate and training
age. Neither question answers the other: a frequent precursor can also occur
frequently without an event, or simply repeat what recent loss already says.

Use the **overview, micro view, episode table and compact forecast table in the
main argument**. If space is tight, move the overview to the experimental
appendix before dropping either quantitative table. The onset-aligned figure
and precision-recall curves support the main claim in supplementary material;
they need not create four large main-text figures. Keep the full eight/nine-row
diagnostic plots as an audit appendix, including absolute norms and actual-update
retention. No conclusion in this section requires all those rows in the main text.

## Figures ready for review

- [Compact overview](../figures/adam_manuscript_overview.pdf): four rows instead
  of six; 7.16 x 4.9 inches instead of 7.16 x 6.8. Loss is raw. Other prominent
  curves use complete, right-aligned eight-observation arithmetic means.
  All raw projected-change values remain visible as faint lines behind the
  mean. Thus an exceedance is not inferred from a smoothed curve.
- [Micro examples](../figures/adam_manuscript_spikes.pdf): the same completed
  largest-relative excursions, now with raw-gradient rho. All points are raw.
  Gradient and stored-momentum norms share one row, each divided by its own
  crop-start value. This compares proportional changes, not absolute magnitudes.
- [Repeated onsets](../figures/adam_episode_onsets.pdf): all 27 large episodes at
  the highest rates, including unfinished recoveries. Individual traces and
  pointwise medians, aligned at the 10% loss-rise onset. No temporal smoothing,
  confidence bands or selection on diagnostic behavior.
- [Forward precision-recall](../figures/adam_forward_precision_recall.pdf):
  previously verified primary forecast, separately displayed within each case.
  Dotted horizontals show positive prevalence. Each case includes all three rates.

All use Analysis 024's three-column, 7.16-inch format, Times New Roman/STIX,
blue/orange/green rate colors, embedded vector fonts, panel letters and light
grids. The new eight-observation display window deliberately differs from
Analysis 024's 50; the visual standard and trailing-mean method are the same.
The previous full diagnostic figures retain their documented smoothing.

## Expanded tables and how to read them

[TABLES.md](TABLES.md) is the readable complete table set. The two main tables
also have generated TeX fragments: [episodes](table_episodes.tex) and
[prediction](table_prediction.tex). Suggested full-width placement uses
footnotesize text and a 3.5pt column separation; avoid shrinking the table as
an image. Explain denominators in captions.

The episode table includes **all nine arms**, episode count, large-event count,
fresh-crossing coverage for all and large events, conditional lead median/range,
gradient- and momentum-growth coverage, and unfinished recovery count. It shows
both recurrence and misses. The highest-rate large-event rows are:

| Case | All episodes | Large | Earlier fresh crossing | Lead median [range] | Gradient growth | Momentum growth |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| MLP | 39 | 17 | 13/17 | 5 [1,8] | 13/17 | 13/17 |
| ResNet-32 | 7 | 3 | 3/3 | 2 [1,3] | 3/3 | 3/3 |
| Transformer | 13 | 7 | 7/7 | 5 [2,8] | 7/7 | 7/7 |

These are warnings 1-8 updates before the 10% onset while loss is still within
5% of its recent minimum. The three MLP 13/17 counts need not identify the same
events. Leads use the earliest qualifying warning in that limited window and
are conditional on a hit; misses are not assigned a zero lead. Large means
observed peak >=50% above the episode baseline. One MLP and one Transformer
episode have unfinished recovery. The all-event crossing counts are 17/39,
5/7 and 13/13: restricting attention to large events improves apparent coverage.

The specificity table keeps **states** as its denominator. A value above 2 has
187 true and 704 false alarm states among 891 alarms: 21.0% precision, 37.8%
recall. A fresh crossing has 64 true among 402 alarms: 15.9% precision. The
future target here is a >=10% rise relative to the current loss within 8 updates;
it is not identical to the episode definition based on a frozen earlier minimum.
Zero-episode low-rate ResNet and Transformer arms still contribute 175 and 190
above 2 alarm states with no positive targets. These are central negative controls.

The forecast table adds **number of decisions, positive targets, prevalence,
average precision, AP difference and Brier score**, per case and pooled. All
models use the same 4,534 decisions, with 495 positives. AP measures ranking
across precision-recall operating points; it is not percent accuracy. Brier
measures squared probability error, not calibration alone. Compare models
within the same row; differing case prevalences make cross-case AP rankings
unhelpful. Case rows subset pooled fits; they do not establish model transfer.

## Strongest supported argument and its qualifications

1. Adding projected change improves primary AP from 0.260 to 0.454 and Brier
   from 0.1197 to 0.0868. Both improve in each of five forward time blocks.
2. Raw cosine alone improves AP to 0.423. Cosine plus gradient growth reaches
   0.433. Projected change is a compact nonlinear combination of those
   ingredients, not a new independent source of geometric information.
3. With all 16 recent log-loss changes in the baseline, AP improves 0.258 to 0.447.
   This reduces the concern that the finding depends only on the three chosen
   loss lags. The richer comparison improves AP in 4/5 blocks, not 5/5; in the
   768-895 block it falls slightly,0.482 to 0.478. Brier improves in all five.
4. With a common larger 48-state exclusion gap, the richer baseline has AP0.244;
   current projected change gives 0.381, versus 0.248 and 0.254 for delays 16 and 32.
   Current timing is more useful in aggregate under this check. In block 768-895,
   delayed features have slightly higher AP than current projected change.
   Delays are a descriptive control, not a valid permutation null or proof
   that training phase has been fully eliminated.
5. More diagnostics are not uniformly better: the original all-feature model
   has pooled AP0.395, below the projected-only extension 0.454. Keep the full
   ablation and nine target/horizon sensitivities, without selecting a winner
   after looking at them.

The loss-only model has worse pooled Brier than a training-prevalence predictor
(0.1197 versus 0.0986). The primary projected extension improves both, but its
ResNet Brier 0.0511 is worse than the constant predictor 0.0376; the Transformer
comparison is 0.0633 versus 0.0618. With the larger gap, even the projected extension
has worse pooled Brier than that reference. The evidence supports incremental
ranking and probability-error improvements over specified loss models, with
calibration/deployment quality unresolved. Do not describe the model as a
validated early-warning system or use a post-hoc operating threshold.

## What reviewers should be able to audit

| Potential objection | Evidence or qualification supplied |
| --- | --- |
| The examples were cherry-picked | Loss-only largest-completed-event rule is explicit; all 129 episodes and misses are retained |
| Aligning at peaks manufactures a lead | Repeat-event figure aligns at 10% onset; warning counts require strictly earlier, still-quiet states |
| There may be a one-step indexing error | All diagnostics use the state where available; raw losses/metrics and the manuscript convention are mapped explicitly |
| Gradient growth is counted twice as evidence | Exact identity links rho, norm ratio and projected change; component comparisons are retained |
| Any oscillation would cross 2 | Quiet-state false alarms and low-rate negative controls are reported |
| The extra feature just repeats loss or training age | Nested comparisons include loss history, age and arm; richer loss history and delayed diagnostics are audited |
| Adjacent steps were randomly split | Common chronological blocks, training-only scaling and exclusion gaps; all labels/inputs reconstructed |
| Thousands of rows masquerade as independent replications | One initialization per case is explicit; no naive IID confidence intervals or confirmatory p-values |
| Pooling hides case differences | Per-case, per-arm and per-block results; ResNet's low positive count is visible |
| A large AP increase proves a useful controller | No operating point, intervention benefit, calibrated risk, curvature threshold or causal mechanism is claimed |

The analysis was motivated by an inspected Transformer peak. Forward blocks
were held out from coefficient fitting, but the study is exploratory and these
blocks are now inspected. Full-batch evaluation removes minibatch sampling from
these trajectories; it does not turn one initialization into independent
replication. Numerical/hardware variation and the limited model/data scope
remain qualifications. No additional seeds are requested for this package.

## Text and methods ready for later integration

- [section_draft.tex](section_draft.tex): proposed scientific narrative, ordered
  context -> micro observation -> recurrence/specificity -> incremental value.
- [methods_draft.tex](methods_draft.tex): exact clocks, reductions, event rules,
  forecasting, sensitivity, uncertainty and scope.
- [captions.tex](captions.tex): proposed captions for all four figures and the
  two main tables.
- [PROTOCOL.md](PROTOCOL.md): choices recorded before the new robustness fits.
- [verification.json](verification.json): source-to-display checks and independent
  refits; raw forecast and summary CSVs are retained beside it.

These files are candidates. They have not been inserted into the manuscript.
The two prior integrated PDFs are preserved under
`../figures/archive/2026-11-02-before-evidence-package/`; the current manuscript
copies remain byte-identical to those versions. Observation 009 and its revision
manifest describe that older integration. Its generator must not be rerun to
produce this new package, and its old source-PDF check is superseded by the
archive mapping in this package's figure manifest. During later integration,
copy the approved new PDFs and update captions/text/checks together.

## Deferred TODO: mechanism

Keep optimizer-state interventions outside the current empirical claim.
Before revisiting them, choose one mechanistic question, a past/current-only
checkpoint rule, paired identical recovery states, a control for adaptive step
length, and fixed-horizon loss outcomes. A specific intervention would test its
own conditional effect. No intervention or launch is prepared or authorized here.

## Reproduction

From the repository root, run scripts 18,19, inspect the four Poppler renders,
then run 19 with `--accept-reviewed` and 20. Script 17 checks the inherited
129-episode/4,534-forecast analysis independently. Do not rerun the historical
manuscript integration generator. All scripts use existing local artifacts.

Method references: [chronological splits and exclusion gaps](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
and [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html).
They define the evaluation tools, not the empirical conclusions of this study.
