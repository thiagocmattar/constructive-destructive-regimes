# Adam state map: exploratory results

The recorded state contains a candidate signal for final paired loss benefit: early adaptive-direction and actual-update accumulation ratios. It is not a validated scheduling rule. The effect is sensitive to the observation window and target, and no tested pre-intervention summary improves primary prediction.

## Target and comparison

Benefit = held-LR validation loss at 128 minus candidate validation loss at 128. Positive is better. Both models, all three seeds, and both altered rates are retained: 12 paired outcomes, six starting states, three seed clusters. The baseline predicts the same model/multiplier cell mean from the other two seeds. Each additional signal gets one slope, fitted after training-only cell centering. The test seed contributes neither cell means nor fitted coefficients.

The early response uses updates 2-16, except accumulation ratios which use the full prefix 1-16. Features are candidate-minus-held values. Initial features use the common pre-update gradient and stored moment, multiplied by log2(LR multiplier). These are predictions after observing both candidate and held responses, not a single-trajectory controller.

## Main comparison

| Predictor beyond model/LR | MAE | RMSE | MAE reduction | Per-seed reductions |
| --- | ---: | ---: | ---: | --- |
| Model/LR only | 0.006643 | 0.008426 | 0% | reference |
| Raw gradient cosine | 0.008686 | 0.011086 | -30.7% | -64.2%, -30.8%, +10.1% |
| Projected gradient change | 0.008166 | 0.009938 | -22.9% | -39.0%, -28.1%, +4.9% |
| Input / prior moment cosine | 0.008527 | 0.011370 | -28.4% | -48.7%, +6.7%, -59.0% |
| Incoming / retained magnitude | 0.008858 | 0.011629 | -33.3% | -85.4%, -17.2%, +4.5% |
| Moment cosine | 0.007630 | 0.009421 | -14.9% | -8.5%, -29.5%, +0.6% |
| Adaptive direction cosine | 0.009878 | 0.012373 | -48.7% | -59.1%, -0.6%, -112.1% |
| Actual update cosine | 0.009901 | 0.012406 | -49.0% | -60.8%, -0.6%, -111.3% |
| Gradient / descent cosine | 0.007312 | 0.009245 | -10.1% | +16.1%, +23.2%, -94.5% |
| Log gradient norm | 0.007181 | 0.010159 | -8.1% | -43.8%, +10.5%, +6.0% |
| Log moment norm | 0.007166 | 0.009369 | -7.9% | -16.8%, -9.7%, +6.0% |
| Log adaptive / moment norm | 0.006409 | 0.008681 | +3.5% | -1.8%, -6.4%, +25.7% |
| Log step norm | 0.012107 | 0.015657 | -82.2% | -12.8%, -37.9%, -237.0% |
| First-order sampled decrease | 0.006509 | 0.007989 | +2.0% | -35.2%, +10.5%, +33.9% |
| Clipping fraction | 0.008826 | 0.011956 | -32.9% | -24.0%, +17.8%, -123.7% |
| Actual displacement / path length | 0.004186 | 0.005098 | +37.0% | +42.4%, +40.8%, +24.4% |
| Raw accumulation ratio | 0.009422 | 0.011581 | -41.8% | -33.5%, -9.5%, -103.1% |
| Moment accumulation ratio | 0.004814 | 0.006386 | +27.5% | -1.7%, +16.4%, +80.8% |
| Adaptive accumulation ratio | 0.004174 | 0.005098 | +37.2% | +42.0%, +41.6%, +24.2% |
| Raw norm growth | 0.008421 | 0.010937 | -26.8% | -63.7%, -16.9%, +2.7% |
| Moment norm growth | 0.010798 | 0.014895 | -62.5% | -188.8%, -8.4%, +5.6% |
| Fixed-probe loss benefit | 0.010238 | 0.012336 | -54.1% | -66.6%, -38.4%, -63.8% |

All candidate features are shown; this is an outcome-known exploratory scan. Adaptive and actual retention are near-duplicate signals, not independent replications. Raw and moment accumulation ratios were added after the first scan as specificity controls.

For actual retention, all three fitted slopes are negative. Conditional on model and LR multiplier, a more negative retention difference predicts greater final benefit in these data. This is not evidence that cancellation should be maximized: the altered LR and evolving trajectory jointly affect geometry and loss.

| Subgroup | Model/LR MAE | + Actual retention MAE |
| --- | ---: | ---: |
| 14M, 0.5x | 0.000857 | 0.000856 |
| 14M, 2x | 0.008717 | 0.006064 |
| 70M, 0.5x | 0.006562 | 0.007346 |
| 70M, 2x | 0.010436 | 0.002478 |

These subgroup errors reveal whether the pooled result is concentrated in a particular intervention; three outcomes per subgroup cannot establish a subgroup-specific prediction law.

## Initial-state prediction

| Common-state feature (action interaction) | MAE | MAE reduction |
| --- | ---: | ---: |
| Raw gradient cosine | 0.010142 | -52.7% |
| Projected gradient change | 0.010000 | -50.5% |
| Log gradient norm | 0.008148 | -22.6% |
| Log prior moment norm | 0.007780 | -17.1% |
| Input / prior moment cosine | 0.009521 | -43.3% |
| Incoming / retained magnitude | 0.008158 | -22.8% |

## Timing and target sensitivity

| Observation cutoff | Actual retention | Adaptive retention | Moment retention |
| ---: | ---: | ---: | ---: |
| 8 | -92.9% | -90.8% | +6.4% |
| 16 | +37.0% | +37.2% | +27.5% |
| 32 | -36.9% | -35.7% | -2.2% |
| 64 | -0.3% | +1.2% | +13.9% |

Entries are pooled MAE reductions versus the same model/LR baseline. The main retention result does not persist across these windows. A separate gradient/descent-angle candidate improves final-benefit MAE at 32, illustrating the multiplicity of this search rather than an independently confirmed discovery.

To separate future improvement from an advantage already realized at 16, the secondary target is benefit at 128 minus benefit at 16. It is validation-based; the predictor uses the distinct fixed training probe. The target itself contains the early loss term, so predictability by early loss need not identify a mechanism.

| Target | Model/LR MAE | + Early probe | + Actual retention | + Both |
| --- | ---: | ---: | ---: | ---: |
| Final benefit | 0.006643 | 0.010238 | 0.004186 | 0.004891 |
| Additional benefit at 16-128 | 0.014465 | 0.008783 | 0.023367 | 0.006710 |

The two-feature combination helps pooled future-benefit prediction relative to the probe comparator, but that conditional improvement is not consistent in every held-seed fold. Thus the standalone final-benefit result does not establish a robust predictor of subsequent recovery. All fold outcomes, coefficients, and candidate combinations are in results.json.

## What the state maps show

The incoming-gradient/prior-moment cosine and their relative contribution magnitudes separate the LR interventions. Color encodes final benefit, not a stability class. This separation alone is largely an action effect: it cannot establish incremental predictive value. Temporal panels show all three seed traces and their mean, with the first 16 updates shaded. Fixed-probe loss is measured only at 0/16/32/64/128; connecting lines are guides, not dense evaluations.

All doubled-rate branches are worse than holding LR on validation at 16, yet better at 128. The state maps describe this adverse early response and later recovery; the shared best action prevents evidence of state-dependent LR selection.

## Limits and next scientific use

Only three seed clusters and one fork stage are available; folds overlap in training data and their scores are not independent replications. A single-slope fit has four training states, and feature/window selection uses the same known cohort. No p-values, population confidence intervals, curvature interpretation, or large-model extrapolation are claimed. The scalar map does not observe the full second-moment spectrum.

The defensible next hypothesis is that early adaptive-update accumulation contains information about the magnitude of the eventual LR response, beyond action and size. Freeze the 16-update feature and target on a new cohort spanning several training stages and both beneficial and harmful LR increases. A policy claim also needs a practical information budget and equal-compute comparison.

## Artifacts

- [Optimizer state map](figures/optimizer_state_map.pdf)
- [Paired temporal responses](figures/paired_optimizer_response.pdf)
- [All primary feature scores](figures/predictive_value_all_features.pdf)
- [Retention prediction and sensitivity](figures/retention_prediction_audit.pdf)
- [Figure sources, reductions, captions, and caveats](observations/001-state-map.md)

Verification and visual review are recorded separately in verification.json and figure_checks.json. No new training or rental.
