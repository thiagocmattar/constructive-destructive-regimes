# Conservative Section VII readability edits

15 November 2026. Each diagnosis and proposed solution was recorded before its source edit. Unlisted prose was retained. Equations, notation, labels, and subsection order are protected.

## Paragraph 1

**Read**

```tex
The displacement formulation distinguishes the magnitude of individual
updates, their local directional interaction, and their accumulated
effect. For plain SGD, the exact decomposition retains interactions at
all temporal lags, while consecutive-gradient cosine provides a local
measurement available during training. The curvature and sampling
relations explain how this measurement changes and why its sign alone
does not determine loss improvement. The full-batch and mini-batch
experiments illustrate these distinctions: gradient reversal can
coexist with continued loss reduction, and weaker reversal need not
produce a better attained loss.
```

**Diagnosis:** The opening leads with abstract terminology and a methods inventory; the key distinction between adjacent opposition and accumulated reinforcement appears late.

**Proposed solution:** Lead with the observed coexistence of opposition, accumulated reinforcement, and loss reduction; retain the exact decomposition versus local measurement distinction in plain language.

**Rewrite:**

```tex
Opposition between consecutive updates can coexist with reinforcement
across the trajectory and continued loss reduction. The displacement
framework distinguishes these effects by accounting for update sizes
and interactions between steps at every separation. For plain SGD,
consecutive-gradient cosine measures only the adjacent interaction.
The curvature and sampling relations explain why its sign alone does
not determine loss improvement. The full-batch and mini-batch experiments
also show that weaker reversal need not yield a lower final loss.
```

## Paragraph 2

**Read**

```tex
The curvature estimate extends this interpretation to the step taken
by any optimizer. Combining its actual parameter change with the
observed gradient difference yields the average directional curvature
along a smooth step, without a Hessian computation. Under Adam, this
step-direction estimate must be distinguished from the combination of
raw-gradient reversal and norm growth. The latter remains useful for
characterizing loss excursions, with timing and missed-event behavior
that vary across models.
```

**Diagnosis:** The any-optimizer claim omits the fixed-objective/full-gradient/nonzero-step conditions. The paragraph conflates retrospective characterization and next-step prediction and does not state the measured diagnostic result.

**Proposed solution:** State the identity conditions and completed-step timing; separate the Adam warning statistic and give the target and comparative result, including the algebraic tie.

**Rewrite:**

```tex
For a smooth, fixed objective, its full gradients at the endpoints of
a nonzero parameter update recover directional curvature averaged along
that step exactly, regardless of the optimizer. This uses gradient and
parameter differences without additional Hessian computations. Under
Adam, the separate combination of raw-gradient reversal and norm growth
does not measure curvature along the applied update. Tested as a warning
of a loss increase of at least $0.1\%$ on the next step, however, it gives
the highest accuracy and warning lift among the evaluated rules in all
three cases, tied with its algebraically equivalent rewriting. These
empirical warnings are distinct from the exact completed-step identity.
```

## Paragraph 3

**Read**

```tex
Following representation changes on the same reference examples makes
the functional expression of these regimes visible. The movement is
concentrated in leading patterns, shares temporal structure across
measured sites, and repeatedly reverses along dominant output
directions. The intervention comparison then tests the usefulness of
controlling its magnitude: all tested displacement-cap quantiles
improve mean training loss and positive variation in the three
continuations, while reversals persist. Endpoint degradation in one
case and additional computation limit
the result to the measured training dynamics.
```

**Diagnosis:** Leading patterns and shared structure compress architecture differences. Improve positive variation is ambiguous, three continuations is unclear, and one case hides the specific endpoint exception.

**Proposed solution:** State the architecture contrast, retain output concentration and temporal structure with measured scope, name the baseline and reduction metrics, and specify ResNet q50 and compute cost.

**Rewrite:**

```tex
Measuring layer responses to the same inputs reveals where oscillation
appears inside the network. In the studied windows, reversal spans all
measured MLP sites; earlier ResNet and Transformer sites reverse less
persistently than their outputs. A few patterns account for most output
movement, with repeated changes in opposite directions and varying
temporal overlap with internal sites. In the matched intervention
comparisons, every tested cap quantile reduces mean training loss and
the sum of upward loss changes relative to unchanged Adam in all three
model--dataset cases. Final loss does not improve uniformly: ResNet q50
finishes above the control. The cap also adds computation, so these
outcomes do not establish faster wall-clock training.
```

## Paragraph 4

**Read**

```tex
The practical contribution is a way to interpret successive gradient
and representation changes together and use that interpretation to
design controlled interventions. Independent initializations,
prospective warning tests, and stochastic or larger-scale training
remain natural next steps. Within the present experiments, the central
lesson is that useful learning does not require removing oscillation;
it requires understanding when the size of the movement becomes
harmful.
```

**Diagnosis:** The last sentence presents understanding harmful movement as a requirement for useful learning, which the experiments do not establish. Future stochastic work should not obscure the existing SGD experiments.

**Proposed solution:** Connect the measurements to intervention design, retain future validation directions with explicit Adam scope, and conclude with the observed improvement despite persistent reversal.

**Rewrite:**

```tex
Together, the framework and measurements connect update interactions
to changes in network representations and motivate interventions on
those changes. Independent initializations, prospective validation of
the warning rules, and stochastic or larger-scale Adam studies remain
next steps. In the studied trajectories, limiting large local
representation changes improved training-loss outcomes while reversals
persisted. These findings support representation displacement as a
target for intervention and show that eliminating reversal is not
necessary for those improvements.
```
