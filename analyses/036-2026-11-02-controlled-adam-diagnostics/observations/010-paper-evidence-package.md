# Observation 010: focused figures and an auditable Adam paper argument

## Request and comparison

Prepare evidence, structure and draft text inside Analysis 036 before another
manuscript edit. Add raw-gradient rho to the compact micro views, reduce the
overview to the same variables, use Analysis 024's figure standard, expand
systematic temporal evidence, and distinguish it from incremental prediction.
Interventions remain a deferred TODO. No new optimizer training, seed, Hessian,
learning-rate change, cloud allocation or spending is introduced.

The evidence package is [paper_evidence/README.md](../paper_evidence/README.md).
Its protocol records added local statistical comparisons before fitting them.
All source training remains the nine completed Run 215 arms: three cases,
three constant rates each, one frozen initialization per case, 1,024 updates.
Matched factors and original coverage remain those of Observations 005 and 008.

## Sources and transformations

The generators are `18_prepare_paper_evidence.py` and
`19_plot_paper_evidence.py`. Inputs include the hash-pinned metrics/boundaries,
Run 215 results/config, full figure manifests, original 129-episode census and
4,534 forward-test predictions. New source and output identities are in
`paper_evidence/evidence_results.json` and `figure_manifest.json`.
Executed training inputs and the previous temporal analysis are unchanged.

Four reviewed, vector-only PDFs use 7.16-inch width, Times New Roman/STIX,
Analysis 024 colors, panel letters, embedded fonts, light grids and no clipping:

| Figure | Reduction and coverage |
| --- | --- |
| `adam_manuscript_overview.pdf` | All nine arms; four rows (loss, raw projected change, raw rho, both relative norms). Loss raw; other prominent lines use complete right-aligned trailing 8 means. All raw projected observations are also drawn faintly. Norms divide by their own initial value. Height 4.9in versus previous 6.8in. |
| `adam_manuscript_spikes.pdf` | Same three loss-selected completed peak windows as Observation 009, now four rows including raw rho. Every point unsmoothed. Gradient/momentum norms divide by their respective crop-start values. |
| `adam_episode_onsets.pdf` | All 27 large highest-rate episodes, including two unfinished recoveries. Raw individual traces and pointwise medians from onset-8 to onset+8; loss divided by frozen event baseline and norms by their own value at-8. No temporal smoothing. Five MLP windows include a previous recovery tail. |
| `adam_forward_precision_recall.pdf` | Original primary forward predictions, separately displayed by case with all three rates; no refit, smoothing or test operating-point selection. Prevalence reference and AP labels. |

All clocks refer to the parameter state where the displayed diagnostics become
available, before the next displacement. Norm normalization represents relative
growth, not absolute gradient/momentum magnitude. Absolute norms and all other
original statistics remain in the full diagnostic figures. The eight-observation
window is a fixed display choice, not a tuned scientific threshold. Every
episode, alarm and forecast uses unsmoothed values. Complete captions are in
`paper_evidence/captions.tex`.

## Systematic evidence

The expanded table includes all nine arms, all/large episode counts, fresh
crossing hit fractions for both severities, conditional warning lead median
and range, gradient/momentum growth hit fractions, misses and recovery censoring.
The CSV retains all five alarms and quartiles. A separate table gives true and
false alarm decisions, precision, recall and false-positive rates on common
forward-test states. It retains zero-positive low-rate ResNet/Transformer arms.
All 129 events,45 large events and original 27 highest-rate large events remain.

At the highest rates, large-event fresh-crossing coverage is 13/17,3/3,7/7;
conditional median leads are 5,2,5 updates with ranges 1-8,1-3,2-8. All-event
coverage is 17/39,5/7,13/13. Warning coverage is not alarm precision; fixed>2
alarm precision is 187/891=21.0%, and fresh-crossing precision is 64/402=15.9%.

The primary forecast table includes decision/positive counts, prevalence,
AP, AP differences and Brier, per case and pooled. It preserves the original
AP0.2602->0.4543 and Brier 0.11972->0.08685, with improvements in all five blocks.

Added, explicitly exploratory checks use the same eligible/test rows and fixed
L2 logistic recipe. Raw cosine alone gives AP0.4227; cosine plus gradient growth
gives 0.4328. A richer baseline with all 16 recent log-loss changes gives
AP0.2580->0.4471 with projected change, and Brier 0.12924->0.09360. AP improves
in 4/5 blocks, with the 768-895 block slightly worse; all five Brier scores improve.

Delay comparisons use a common larger 48-state gap to protect the earliest
delayed input. AP is 0.2436 for richer loss,0.3815 with current projected change,
0.2475 with its 16-state delay and 0.2542 with its 32-state delay. Current timing is
more informative in aggregate; it does not win every block. This control is
not a permutation null, a causal test or proof against all phase confounding.

The training-prevalence probability reference reveals an important qualification:
the original loss baseline has worse pooled Brier (0.11972) than the constant
reference (0.09863). The primary projected extension improves both, but its
ResNet/Transformer Brier does not beat that reference; with the larger gap,
its pooled Brier is also worse than the constant reference. Ranking value and
incremental improvement do not establish calibrated operational risk estimates.
Complete ablations, all original horizon/rise sensitivities and block/case
heterogeneity are retained, including unfavorable outcomes.

## Argument, interpretation and uncertainty

The proposed paper structure uses the micro view to explain the measurement,
the episode census to assess recurrence/specificity, and forward prediction to
test incremental information beyond recent loss. These are different empirical
questions, not redundant demonstrations or independent replications.
The overview establishes coverage; additional onset/precision-recall figures
can be supplementary. Draft section, appendix methods, captions, generated TeX
tables and an explicit reviewer-objection/evidence map are in `paper_evidence/`.

The exact projected-change identity couples raw reversal and norm growth.
Their simultaneous presence is not three independent mechanisms. Above 2 is
not an Adam curvature estimate or calibrated stability boundary. Event alignment
conditions on future loss; all eligible quiet states are needed to assess
prediction and false alarms. Events and forecast windows are dependent, rates
share initializations, and the original inspected peak motivated the analysis.
There are no IID confidence intervals, confirmatory p-values, independent-seed
robustness claims, task-transfer claims or intervention-effect claims.

## Verification and preservation

`20_verify_paper_evidence.py` independently checks 90 episode-summary rows,
45 alarm-summary rows,9,068 new robustness prediction rows,30 richer/delayed
fits, all 225 plotted curves and 67,530 values against original histories.
Independent NumPy-standardized refits reproduce probabilities exactly;
maximum curve discrepancy is 1.42e-14 from arithmetic mean rounding. It checks
availability clocks, target coverage, temporal gaps, within-case/block scores,
raw source hashes, embedded fonts, vector-only PDFs and current visual acceptance.
`verification.json` records the final result. Script 17 separately rechecks the
inherited census and primary forecasts. All four Poppler renders were reviewed.
The candidate section, appendix methods, captions and both generated tables
also compile in an isolated six-page TeX review document, without errors or
overfull boxes. The table page was rendered and inspected at the intended
7.16-inch text width; this scratch review is not a manuscript build.

The old integrated overview/spike PDFs are preserved under
`figures/archive/2026-11-02-before-evidence-package/` and still match the active
manuscript copies byte-for-byte. Observation 009 and its revision manifest
describe those older versions. This task changes Analysis 036 only; no manuscript
text or copied figure is replaced. Old revision checks that demand its source
PDFs at the now-updated analysis paths are historical and must use this archive
mapping until an approved integration updates the manuscript and its checks.
Interventions remain a future scientific-design TODO, with no launch prepared.
