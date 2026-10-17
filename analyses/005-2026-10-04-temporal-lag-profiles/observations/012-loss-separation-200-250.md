# Observation012: loss separation around updates200-250

The user asked what happens at updates200-250 that allows the larger model to
converge faster. This local descriptive check uses the same Pythia-14M/31M/70M
measurement replays, Runs017/018/019 and seed1235, as observations010/011. It adds
no intervention, seed, training or figure. Data/input identities, model/data seed,
optimizer recipe, LR histories, token coverage and accepted artifact hashes are
checked using the preceding figure manifest. Parameter count, architecture and
initial tensor values differ across sizes. This is an exploratory interval chosen
after viewing the curves, not a predeclared change-point test.

## Main observation

The 70M model largely maintains its previous loss-reduction rate at200-250 while
the smaller models slow. Endpoint recorded-training-loss reductions are:

| Model | Updates150-200 | Updates200-250 | Updates250-300 |
| --- | ---: | ---: | ---: |
| 14M | 4.915% | 2.804% | 2.008% |
| 31M | 4.965% | 2.650% | 2.468% |
| 70M | 5.298% | 5.097% | 6.203% |

Reduction is 100*(1-L_last/L_first), using unsmoothed losses at the endpoints,
not an average of overlapping N=21 plot values. Each interval has50 changes and
51 loss records. OLS slopes of ln(loss) across all51 records support the pattern:

| Model | Updates150-200 | Updates200-250 | Updates250-300 |
| --- | ---: | ---: | ---: |
| 14M | -0.0009423 | -0.0005895 | -0.0004734 |
| 31M | -0.0009477 | -0.0006009 | -0.0005306 |
| 70M | -0.0010798 | -0.0010311 | -0.0014175 |

The fitted70M decline is therefore slightly slower at200-250 than150-200. Its
further steepening appears at250-300. Different minibatches and their composition
can affect these endpoint/trend summaries; the exact onset is not established.

## What changes in the available diagnostics

The geometry values below are descriptive arithmetic means of the original
per-update scalars across the same inclusive51-record windows. The figure remains
unsmoothed for geometry.

| 70M quantity | Updates150-200 | Updates200-250 |
| --- | ---: | ---: |
| Direction L2 norm | 2092.04 | 2023.36 |
| Adjacent direction cosine rho_d | 0.92466 | 0.91866 |
| Norm of adaptive component eta*d | 1.81999 | 1.59361 |
| Raw gradient L2 norm | 0.46263 | 0.49527 |

There is no direction-norm surge or improved adjacent direction alignment in
this comparison. At200-250,70M's mean raw gradient norm0.49527 is smaller than
14M's0.53294 and31M's0.54436, although its adaptive direction norm is larger
(2023.36 versus1359.16 and1629.35). Adam preconditioning and the different number
of parameters prevent interpreting direction-norm ordering as gradient strength.
eta*d excludes decay and floating-point subtraction effects; it is not the full
recorded parameter displacement.

Every size has the identical continuously declining LR:0.000831251 at200 and
0.000741263 at250. All51 updates in this interval have full1024-sequence batches,
2096128 supervised tokens, no clipping activation, no overflow and loss scale4096.
The frozen data protocol uses one seeded permutation shared across sizes, with
no scheduled data phase or optimizer change at200. The event streams contain only
initial validation, accepted updates and final validation. These checks do not
identify the local minibatches' semantic difficulty for each model.

## Mechanism remains unresolved

Adjacent direction cosine describes persistence of the optimizer direction.
It does not measure whether that direction reduces the objective efficiently.
For a twice-differentiable fixed objective F and an adaptive-only step -eta*d,
the first-order loss change is -eta*<gradient(F),d>, with a directional-curvature
remainder. Full AdamW adds the decay displacement. Neither the needed
gradient-direction inner product nor that curvature is determined by ||d||,
rho_d, q_d or raw-gradient norms. A lower adjacent rho_d is not evidence of a
better or worse descent direction. The logged adjacent batch losses also
change the evaluation minibatch, so they do not isolate a same-objective step.

The retained streams do not include per-update gradient-direction inner products
or same-batch before/after losses. Initial/midpoint/final checkpoints cannot
supply those missing histories. Fixed-probe loss, gradient-direction inner
products and same-batch loss before/after updates around this interval would
help distinguish useful direction changes, curvature/conditioning and minibatch
effects. No new run is authorized or launched by this analysis. One seed and
post-hoc window selection provide no causal capacity, formal convergence,
stability or generalization conclusion; no seed uncertainty interval is claimed.

## Reproduction and checks

Run `python 17_inspect_loss_separation.py`. The
[reduction](../loss_separation_200_250.json) records the source manifest/hash,
accepted source hashes, all five exploratory50-update intervals from100-350,
endpoint losses, log slopes, geometry means, clipping counts, loss scales and LR.
OLS is checked independently against numpy.polyfit; log differences telescope
to the endpoint log ratio. Input identities and full LR histories agree exactly
across sizes, LR decreases throughout150-350, accepted updates are complete,
and every existing PDF is unchanged by SHA-256.
