# Current claim and evidence audit

- **Updated:** 8 November 2026; aligned with the four contributions in [PLAN.md](../PLAN.md).
- **Clock:** state s means s completed updates. Raw gradients at s describe the incoming response; representation change s→s+1 is observed after the outgoing update.
- **Mathematics:** [III equations](03-regimes/equations.md), [Adam response](05-adam/a-characterization/equations.md), [tracing](05-adam/b-tracing/equations.md), [cap and loss accounting](05-adam/c-intervention/equations.md).
- **Verification scope:** copied artifacts are hash checked; compact tables are reconstructed from retained reductions. This packaging pass does not rerun every historical scientific computation.

| Claim | Verdict / precise boundary | Local evidence |
|---|---|---|
| Adjacent destruction implies negative cumulative interaction | Rejected: all nine GD total pair interactions are positive; seven adjacent sums are negative. | [All-lag observation](04-gd-sgd/observations/56-001-paired-displacement.md) |
| The proxy provides inexpensive directional Hessian information | Supported for smooth GD as an incoming path average; point curvature needs the error bound. Reciprocal BB1 identity requires explicit credit. | [Equations and assumptions](03-regimes/equations.md) |
| Noise-corrected proxy independently matches directional derivatives | Supported by 1,188 HVP pairs in the tanh MLP; batch-32 median discrepancy .7467→.0030. Dependent observations, not 1,188 independent experiments; correction uses offline full gradients. | [Noise observation](04-gd-sgd/observations/27-001-controlled-minibatch-noise.md) |
| The quadratic loss condition predicts every nonlinear update | Rejected: highest-LR Transformer has 1,325/2,999 sign disagreements. | [Finite-step audit](04-gd-sgd/observations/24-007-section-iv-critical-review.md) |
| The same raw-gradient computation transfers to Adam | Supported as a mixed Hessian response under a fixed smooth objective. It is not actual-update alignment or a universal Adam stability boundary. | [Adam equations](05-adam/a-characterization/equations.md) |
| Q≥2 precedes every next-step increase | Rejected: high-LR coverage is 172/454, 21/95, 81/152; it also precedes decreases. | [Complete sign table](05-adam/a-characterization/tables/next-step-signs.md) |
| Severity and forward timing are informative | Supported descriptively: at ≥10% the gradient-sign/norm conditions differ; Transformer Q coverage is 22/26, 25/26, 26/26 at leads 1,2,3. False alarms and MLP counterexamples remain. | [Severity table](05-adam/a-characterization/tables/severity-10pct.md), [audit](05-adam/a-characterization/observations/55-004-timing-severity-audit.md) |
| Current audit demonstrates added predictive value | Not established. One seed/model, dependent states, inspected thresholds/horizons. A held-out forecasting comparison remains conditional. | [Existing held-out-rate limits](05-adam/a-characterization/observations/57-001-response-memory.md) |
| One stationary mode explains all networks | Too strong: output rank95 is 5/16/15 over 51/31/49 fitted rows. Concentration and temporal alternation are distinct. | [Event-window report](05-adam/b-tracing/observations/63-005-technical-report-event-window.md) |
| Concentration is explained only by a few large steps | Inconsistent with retained-Gram controls: directional overlap adds concentration beyond identical-norm orthogonal references. Unit-row rank95 5/18/29 exposes amplitude dependence. | [Spectral table](05-adam/b-tracing/tables/spectral-controls.md) |
| Tracing identifies the unique initiating layer | Unproven. Stage participation differs: first-projection MLP motion, deeper ResNet agreement, and output-like Transformer reversal after attention. | [Tracing method](05-adam/b-tracing/observations/63-METHOD.md) |
| Gradient/output correlation is universally >.7 | Rejected: MLP's 450–500 fit is .642; .776 belongs to the longer fit. | [Original report table](05-adam/b-tracing/report/results-table.tex) |
| Removing reversals improves training | Rejected as a rule: directional removal and freezing can worsen mean loss; effective caps retain reversals. | [Negative controls](05-adam/c-intervention/observations/65-009-what-the-negative-controls-show.md) |
| Local caps improve selected training windows | Supported: −22.85/−16.22/−26.00% mean loss, with two further MLP seeds −19.43/−32.08%. | [Paired outcome table](05-adam/c-intervention/tables/distributed-cap-outcomes.md) |
| Local caps bound total output motion or loss | False. They bound direct affine RMS per targeted unit on the current probe inputs; incoming motion and interaction are uncapped. | [Cap equations](05-adam/c-intervention/equations.md) |
| Representation-sensitive allocation adds value beyond a layer norm schedule | Supported in the MLP matched comparison: −22.85% versus −3.43% mean loss. The control uses a retrospective schedule. | [Negative controls](05-adam/c-intervention/observations/65-009-what-the-negative-controls-show.md) |
| Increasing LR under a cap accelerates learning | Not supported by the 54-condition MLP grid: each quantile's across-seed mean worsens at 2×/4× versus its own 1×. | [Quantile/LR observation](05-adam/c-intervention/observations/65-013-distributed-cap-quantile-lr.md) |
| Training-window benefit implies generalization benefit | Rejected for reported terminal comparisons: test loss rises 2.87/8.18/100.17% in the three seed-0 cases. Endpoints are not held-out trajectories. | [Outcome table](05-adam/c-intervention/tables/distributed-cap-outcomes.md) |
| Lower beta2, clipping or lower LR generally fail | Unsupported blanket statements. Effects depend on context; a late beta2 switch is not lower beta2 from initialization; some clipping is inactive. | [V-C comparator records](05-adam/c-intervention/README.md) |

## Analytic counterexamples retained

- A mixed-projection value above two need not imply a loss increase for arbitrary updates: H=diag(100,1), theta=(.01,1), eta=1, d=(.05,.5) gives Q=2.75 while loss falls .505→.205.
- The reverse also fails: H=I, theta=(1,0), d=(0,2) gives Q=0 while loss rises .5→2.5.
- These are exact quadratic counterexamples to transplanting the GD threshold; they are not Adam experiments.
- Exact MLP output accounting shows all 377/371/379 baseline rises >1e−8 have negative residual correction exceeded by squared movement over outgoing 448–1152, including the extra transition to 1153. This exact identity does not transfer unchanged to cross-entropy.
