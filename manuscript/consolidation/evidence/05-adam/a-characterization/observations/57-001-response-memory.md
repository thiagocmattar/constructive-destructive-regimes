> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [001-response-memory.md](../../../../../../analyses/057-2026-11-05-adam-response-memory/observations/001-response-memory.md).

# Observation001: event specificity and the Adam memory connection

Source: frozen Run215 nine-condition, 1,024-update cohort, authenticated through
Analysis036's manifest. One seed per architecture, matched across learning rates;
hardware varies across some original conditions. Analysis055 supplies the separate
ordinary-trajectory control. No new training or Hessian measurement was executed.

`01_analyze.py` follows the pre-reduction contract in README.md. Independent raw
loss segmentation, QR regression fits, optimizer dot-product signs and mathematical
checks are in `02_verify.py` / verification.json. All 9,225 loss states and 9,216
outgoing updates are covered. Source paths and hashes are in results.json.

## Findings

- Strict 5/10/20% cutoffs select 206/129/74 recovered excursions, respectively;
  each also retains two censored candidates. These are dependent events, not
  independent replications. The 10% sample has 127 defined pre-rise cosines.
  The middle/high MLP arms supply 84 of those observations.
- All 794 outgoing updates in the 129 recovered rise intervals have negative
  raw-gradient dot actual parameter increment. Recovery has 1,374/1,376 negative
  products. All seven event-weighted mean gradient/direction cosines are positive
  in both phases. A finite step can increase loss despite a descending first-order
  contribution. This does not identify a unique mechanism for the loss increase.
- The additive cosine term improves log-norm within-run R-squared by
  0.005764--0.220731. It worsens held-out-rate MSE in all seven folds, whereas in
  the linear-norm family it worsens five of seven. The interaction can improve
  some held-out comparisons; no specification gives uniform transfer. All 112
  fits, coefficients, counts and errors are retained, including negative R-squared.
- The additive cosine coefficient changes sign: positive for the lowest-rate
  MLP and middle-rate Transformer, negative for the other five trajectories in
  both norm families. Association is conditional on the selected population and
  transformation, not a demonstrated universal role for alignment.
- Leaving out a rate is not leaving out an initialization. Only two populated
  rates exist for ResNet/Transformer, with test counts as low as six. Poor transfer
  can reflect different event support; no independent-seed uncertainty is inferred.
- The proposed response/memory algebra passes 1,200 rotated-quadratic step checks,
  100 smooth nonquadratic cases, 396 memory-recursion checks and 400 exact-alternation
  checks. Maximum scaled quadratic discrepancy is 7.61e-14. These are mathematical
  verification cases, not empirical neural calibration. ReLU path boundaries may
  violate the regularity needed for the classical-Hessian integral identity.

## Figures and table

Generator: `03_plot.py`. Source rows are event_features.csv and regressions.json;
the table uses event_incidence.csv. figure_manifest.json records every plotted
coordinate, its hash, the PDF hashes and manuscript-copy identity. No smoothing.

1. **adam_severity_product_supported.pdf.** One point for each of 127 recovered
   >10% events with defined pre-rise cosine. Predictor rho*A; response is peak /
   pre-rise loss on a log axis. Separate OLS fits on log response stop at each
   rate's observed predictor range. Seven rate groups; no extrapolated lines.
   This replaces the compiled scatter, preserving the original Analysis036 PDF.
2. **adam_severity_ablation.pdf.** Thirty-five points: five predictor specifications
   for each of seven eligible trajectories, on identical event samples. All fits
   include an intercept. The additive/interaction models have more coefficients,
   so in-sample improvement alone does not establish useful additional information.
   Full linear-norm alternatives and held-out-rate results remain in the tables.
3. **analysis057_incidence.tex.** Nine complete trajectories; recovered counts at
   three thresholds, censored >10% candidates, occupancy, pre-rise reversal counts
   and absolute peak increases. Occupancy is defined on outgoing update intervals
   and includes censored candidates. Objective units differ by model, so absolute
   increases are not pooled across architectures.

No error bars are supplied: one initialization per model cannot yield meaningful
independent-seed uncertainty. Phase means weight events equally; local-descent
counts weight each outgoing update once. Missing initial cosine stays undefined.

## What remains

Analysis055's fixed eight-update outcome supplies useful ordinary-trajectory
controls, but is not an event-duration-matched experiment. Independent seeds,
duration-matched controls, same-state HVP comparisons, controlled optimizer-state
branches and stochastic/larger-model monitoring remain in the revision TODO and
experiment plan. None is claimed as completed by the revised paper.
