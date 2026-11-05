# Analysis057: Adam response, memory and event specificity

Authorized by the 5 November Adam revision request. This analysis reuses the
nine frozen Run215 trajectories and Analysis036 loss-defined events. It supports
the manuscript revision and plans, but does not launch, further training.
The supplied notes and proposed TeX remain unchanged in the revision folder.

## Measurement contract (fixed before reduction)

- Population: all nine 1,024-update full-objective Adam trajectories, one seed
  per model and three rates. Preserve Analysis036's fixed-pre-rise detector,
  including its six censored final excursions. All 9,225 loss states remain.
- Report strict peak increases above 5%, 10% and 20%, recovered/censored counts,
  absolute increases, durations, positive loss-excess sums, and update occupancy.
  Occupancy counts the union of outgoing update indices a..end-1, divided by
  1,024; include censored events in incidence, exclude them from recovery means.
- Primary severity sample: recovered >10% events with defined pre-rise cosine.
  Outcome is ln(peak/pre-rise loss). A is the peak/pre-rise raw-gradient norm
  ratio. OLS with an intercept compares rho, A, rho+A, rho+A+rho*A, rho*A,
  ln(A), rho+ln(A), and rho+ln(A)+rho*ln(A). Compare alignment additions within
  each transformation family. Keep raw coefficients and training rank. These
  are descriptive regressions, not causal effects or online forecasts.
- Fit within each model/rate and leave one complete rate trajectory out within
  each model. A held-out rate shares its seed with training rates: this checks
  transfer across rates, not independent initialization replication. Require
  >=5 training events and full design rank; record unsupported fits explicitly.
  Report held-out mean squared error and R-squared relative to the held-out
  mean; negative values remain visible. No hyperparameter selection or p-values.
- Cross-alignments use retained whole-parameter raw-gradient/current-moment
  and raw-gradient/adaptive-direction products. For actual local descent use
  recorded dot(g, applied_delta), including FP32 update rounding. Event rise
  uses outgoing states a..p-1; recovery uses p..c-1. Summaries average states
  within each event, then weight events equally. Report counts per trajectory.
- The exact first-moment residual recurrence and response-curvature identity
  receive independent FP64 mathematical checks on synthetic vectors/quadratics
  and smooth nonquadratic functions. These are algebra/implementation checks,
  not new neural-network evidence or empirical validation of the approximation.
- Zero denominators are undefined, never epsilon-filled. No smoothing or
  fabricated uncertainty; events within a trajectory are dependent. FP64
  scalar reductions of retained FP32-run measurements preserve source precision.
- Analysis055 supplies separate ordinary-trajectory context comparisons. Its
  fixed eight-update future marker is not the event-duration-matched control
  requested in the new notes. That remaining comparison stays in the TODO.

Every derived result pins source hashes. Publication figures and their numeric
inputs have an observation with coverage, reduction, caption and caveats.
New independent seeds, HVPs on neural checkpoints, optimizer-state branches,
stochastic monitoring and larger-model benchmarks remain proposed experiments.
