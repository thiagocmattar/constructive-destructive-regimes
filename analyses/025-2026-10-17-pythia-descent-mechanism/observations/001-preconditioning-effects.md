# Observation001: norm-controlled preconditioning interventions

Sources: all27 local retrieval receipts and SHA-256 inventories, listed in
`results.json` provenance; frozen cohort/configs/probe; raw `events.jsonl`,
`probe_events.jsonl`, and `summary.json` retained under each owning run.
Generating script: `python analyses/025-2026-10-17-pythia-descent-mechanism/08_analyze.py`.

Coverage:9 baselines (14M/31M/70M, three seed blocks,350 complete updates)
and18 matched50-update branches from175/225. Every baseline has9 probe points
and seven same-state candidates per point. The same128-block fixed probe is
disjoint from all training prefixes; it is a held-out portion of the training
corpus, not a separate validation population. Full original validation is retained
in each run's summary. Reduction is token weighted, with no display smoothing.

The primary effect is tau = ln F_ablated(end) - ln F_native(end), at identical
calendar update and common fork. Positive tau means removing preconditioning
increased probe loss. Adaptive norms follow the unmodified parent's schedule.
The per-seed70M-minus14M and31M-minus14M effects are in `size_interactions`.

| Fork | Size | Mean tau | Minimum seed tau | Maximum seed tau |
| --- | --- | ---: | ---: | ---: |
| 175 | 14M | 0.208094 | 0.181416 | 0.233604 |
| 175 | 31M | 0.350102 | 0.320622 | 0.399874 |
| 175 | 70M | 0.522908 | 0.486341 | 0.562303 |
| 225 | 14M | 0.23775 | 0.200716 | 0.265057 |
| 225 | 31M | 0.456766 | 0.415563 | 0.50459 |
| 225 | 70M | 0.725101 | 0.660009 | 0.778978 |

Figure captions:

- `native_descent_accounting.pdf`: fixed-probe log-loss change and actual
  one-step D, linear prediction P, nonlinear remainder R, with D=P-R. Each
  decrease is normalized by that model's pre-step probe loss. Thin curves show
  seeds; thick curves show arithmetic means at measured updates. Segments join
  sparse measurements and do not represent intermediate probe evaluations.
- `preconditioning_effects.pdf`: primary50-update effects after each fork;
  points show all seed blocks, horizontal marks show means, grey segments pair
  the same seed across sizes. No uncertainty interval is inferred from updates.
- `direction_substitutions.pdf`: one-step relative loss decreases at175,200,225,
  250; small dots show three seeds, large dots show their mean. Direction
  substitutions match native adaptive norm within each model; half/double
  controls deliberately change it. Native/reconstructed Adam are numeric controls.

Uncertainty and interpretation: only three seed blocks, with initialization and
data ordering co-varying within a block. Report seed effects rather than treating
updates as replicates. The motivating200–250 interval was selected post hoc.
Within-model interventions identify an optimizer-mediated contribution under
this recipe; cross-size differences do not identify a causal capacity effect.
Adaptive norms are controlled before FP32 rounding; full displacement also has
decay, whose vector changes once trajectories diverge. R includes higher-order
terms and numeric error and is not a curvature measurement. One-step changes
hold state fixed and cannot alone establish a sustained convergence mechanism.
All three PDFs were rendered and visually inspected at closeout; raw metrics remain local.

Scientific interpretation and completed cloud/billing records: [REPORT.md](../REPORT.md).
