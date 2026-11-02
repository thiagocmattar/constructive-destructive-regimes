# Exact long-lag gradient profiles

Additional requested view: [largest GD rate, consecutive lags 1–256](figures/largest_lr_lag256.pdf).
It uses MLP LR 0.04, ResNet-32 LR 0.1 and Transformer LR 0.2, with all three
anchor rows and one ordered curve per panel. [Sources and caption](observations/002-largest-lr-lag256.md).

Completed 27 profiles / 36,096 anchored cosines across the nine
[Analysis024](../024-2026-10-17-eos-gd-lr-calibration/README.md) conditions,
at anchors 500/2000/3000. All profiles reach lag 256;
the largest measured lag is 2048. There are 25 primary
profiles and two separately flagged historical Run062 reconstructions.
The original starts comprise 21 direct saved starts, four reconstructions that
passed the prior historical checks, and two that failed. This follow-up reuses
their retained Run205 states; it does not replay initialization again.

Among the 17 primary profiles that originally reverse at lag 1,
4 have a terminal local half-gap at most 0.1, and 13 remain above
it at their final measured horizon. Of these, 4 stopped at a
declared stage because the criterion was met. The score is a median half absolute odd/even
cosine gap of at most 0.1 over the final 32 adjacent pairs. It describes the
alternating component of this anchored curve; it does not require the cosine's
common offset to vanish. Persistent curves are retained as observed.
Cosine normalizes gradient magnitudes: shrinking gradient norms can coexist
with persistent angular alternation. This score does not measure decay of the
gradient norm or the amplitude of parameter oscillations.
A small anchored half-gap also does not show that consecutive gradients have
stopped reversing, even within the measured range.

For example, the early Transformer at LR 0.2 has local half-gap 0.225 at lag 512, rising to 0.600 at 1024 and ending at 0.326 at 2048. More lags reveal a rebound, not monotonic damping.

- MLP: 3 primary initially reversing profiles; 0 stage-criterion stops, 3 terminal gaps above 0.1; terminal gap range 0.173–0.259.
- ResNet: 8 primary initially reversing profiles; 2 stage-criterion stops, 6 terminal gaps above 0.1; terminal gap range 0.070–0.518.
- Transformer: 6 primary initially reversing profiles; 2 stage-criterion stops, 4 terminal gaps above 0.1; terminal gap range 0.038–0.645.

These endpoint ranges have unequal horizons; they are descriptive summaries,
not matched-coverage model or LR rankings. The common-range figure supports
comparisons at the same measured lags.

- [Common lag 256 figure](figures/common_lag256.pdf)
- [Longer primary profiles](figures/long_lag_profiles.pdf)
- [All profiles with reconstruction flags](figures/all_profiles_with_reconstruction_flags.pdf)
- [Local odd/even half-gap](figures/alternating_gap.pdf)
- [Every value](lag_profiles.csv), [verified results](results.json), [verification](verification.json)
- [Observation and captions](observations/001-long-lag-profiles.md)
- [Execution closeout](closeout.json), [summary and timing](summary.json)

## Coverage and interpretation

Final horizons: 9 at lag 256, 2 at lag 512, 16 at lag 2048. Positive-lag 1 controls stop at 256. Initially
negative lag-1 profiles extend through the frozen stages 256/512/1024/2048,
stopping at the first stage meeting the envelope threshold. There were
0 budget/deadline-limited profiles. Coverage beyond 256 is
adaptive and outcome dependent; compare all conditions on the shared 256-lag
figure, and use the longer figure to inspect persistent trajectories.

The half-gap companion plots the declared local stopping statistic over every
complete 32-pair window. A threshold crossing is local: some profiles first
attenuate and then rebound. It does not establish asymptotic decay or guarantee
that unmeasured future lags remain small. Complete odd/even pairs are used;
an unmatched final odd lag is retained in the raw curve and omitted from this score.

The plots join measured even lags and odd lags separately, without smoothing,
centering or projection. Each value is one whole-model anchored pair,
cos(gradL(theta_anchor),gradL(theta_(anchor+k))), rather than an average over
many training positions. Late windows continue beyond the original 3000-state
history. All new forks preserve original LRs, plain SGD, seed 0, model and full
training objective. They do not establish seed uncertainty, stationarity,
stability, generalization or a loss benefit from alignment.

| Task | Parent | LR | Anchor | Eligibility | GPU | Last lag | Cosine 256 | Final cosine | Gap 256 | Final gap | Stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 059 | 0.004 | 500 | primary | 5090 | 256 | 0.968664 | 0.968664 | 0.000088 | 0.000088 | positive_lag1_control_minimum_horizon |
| mlp | 059 | 0.004 | 2000 | primary | 5090 | 256 | 0.996440 | 0.996440 | 0.000012 | 0.000012 | positive_lag1_control_minimum_horizon |
| mlp | 059 | 0.004 | 3000 | primary | 5090 | 256 | 0.998085 | 0.998085 | 0.000006 | 0.000006 | positive_lag1_control_minimum_horizon |
| mlp | 062 | 0.02 | 500 | primary | 5090 | 256 | 0.953085 | 0.953085 | 0.000143 | 0.000143 | positive_lag1_control_minimum_horizon |
| mlp | 062 | 0.02 | 2000 | unqualified replay | 5090 | 256 | 0.327357 | 0.327357 | 0.079976 | 0.079976 | alternating_envelope_below_threshold |
| mlp | 062 | 0.02 | 3000 | unqualified replay | 5090 | 2048 | -0.126620 | -0.013306 | 0.106077 | 0.361377 | maximum_horizon_reached |
| mlp | 122 | 0.04 | 500 | primary | 5090 | 2048 | 0.092305 | -0.025773 | 0.146560 | 0.173413 | maximum_horizon_reached |
| mlp | 122 | 0.04 | 2000 | primary | 5090 | 2048 | 0.260091 | 0.137579 | 0.187064 | 0.258982 | maximum_horizon_reached |
| mlp | 122 | 0.04 | 3000 | primary | 5090 | 2048 | 0.039003 | 0.483712 | 0.398542 | 0.194774 | maximum_horizon_reached |
| resnet | 123 | 0.025 | 500 | primary | 5090 | 256 | 0.854716 | 0.854716 | 0.000465 | 0.000465 | positive_lag1_control_minimum_horizon |
| resnet | 123 | 0.025 | 2000 | primary | 4090 | 2048 | 0.909285 | -0.382839 | 0.905639 | 0.518149 | maximum_horizon_reached |
| resnet | 123 | 0.025 | 3000 | primary | 4090 | 2048 | -0.861117 | 0.225513 | 0.524005 | 0.223325 | maximum_horizon_reached |
| resnet | 124 | 0.05 | 500 | primary | 5090 | 2048 | 0.598743 | 0.151009 | 0.628639 | 0.119935 | maximum_horizon_reached |
| resnet | 124 | 0.05 | 2000 | primary | 4090 | 2048 | 0.386606 | -0.302790 | 0.208906 | 0.127930 | maximum_horizon_reached |
| resnet | 124 | 0.05 | 3000 | primary | 4090 | 2048 | -0.291114 | 0.098273 | 0.460202 | 0.335668 | maximum_horizon_reached |
| resnet | 120 | 0.1 | 500 | primary | 4090 | 2048 | 0.497498 | -0.049178 | 0.443023 | 0.069807 | alternating_envelope_below_threshold |
| resnet | 120 | 0.1 | 2000 | primary | 4090 | 2048 | -0.167915 | 0.148411 | 0.319238 | 0.113005 | maximum_horizon_reached |
| resnet | 120 | 0.1 | 3000 | primary | 4090 | 2048 | -0.072311 | -0.147249 | 0.215453 | 0.097894 | alternating_envelope_below_threshold |
| transformer | 121 | 0.05 | 500 | primary | 4090 | 256 | 0.937513 | 0.937513 | 0.000188 | 0.000188 | positive_lag1_control_minimum_horizon |
| transformer | 121 | 0.05 | 2000 | primary | 4090 | 256 | 0.988167 | 0.988167 | 0.000037 | 0.000037 | positive_lag1_control_minimum_horizon |
| transformer | 121 | 0.05 | 3000 | primary | 4090 | 256 | 0.991473 | 0.991473 | 0.000028 | 0.000028 | positive_lag1_control_minimum_horizon |
| transformer | 125 | 0.1 | 500 | primary | 5090 | 2048 | 0.955937 | -0.569430 | 0.937482 | 0.645209 | maximum_horizon_reached |
| transformer | 125 | 0.1 | 2000 | primary | 5090 | 2048 | -0.552122 | 0.186283 | 0.300098 | 0.475088 | maximum_horizon_reached |
| transformer | 125 | 0.1 | 3000 | primary | 5090 | 2048 | -0.935148 | 0.861196 | 0.570986 | 0.607093 | maximum_horizon_reached |
| transformer | 126 | 0.2 | 500 | primary | 5090 | 2048 | -0.964119 | 0.828424 | 0.965433 | 0.325628 | maximum_horizon_reached |
| transformer | 126 | 0.2 | 2000 | primary | 5090 | 512 | 0.186109 | -0.137846 | 0.257997 | 0.041304 | alternating_envelope_below_threshold |
| transformer | 126 | 0.2 | 3000 | primary | 5090 | 512 | 0.559515 | 0.206230 | 0.619917 | 0.037867 | alternating_envelope_below_threshold |

The GPU column identifies final execution hardware. Some handed-off prefixes
were measured on another GPU; source paths and activation records retain that
provenance. Canonical final profiles by source run: {'206': 20, '207': 3, '208': 4}.

## Exact streaming and numerical qualification

No compression was required: the fixed anchor and current/recent full gradients
bound memory independently of horizon. Recorded peak CUDA allocation is
10.278 GB and peak reservation is
14.898 GB. Parameters and accumulated gradients
are FP32, TF32 is disabled, and diagnostic products/norms are FP64. Every new
lag was checked using independent NumPy CPU and CUDA reductions. Maximum cosine
disagreement is 2.89e-15; all 95 retained
full-gradient vector audits pass. The 297 source rows, including all 27 anchors,
are unchanged. Zero denominators are explicitly undefined.

The original full training coverage is unchanged: the first 5,000 CIFAR MLP
examples with half squared error summed over 10 outputs and averaged over examples;
50,000 ResNet examples with 50 fixed 1,000-image train-mode BN groups; and 59,680
Transformer sequences of length 35 with 2,088,600 supervised targets and token-weighted NLL.
Existing -100 padding is excluded. No held-out evaluation was run. Original source metadata
and per-file hashes are retained under Run206.

Calibration selected a common Transformer compute chunk of
512 from 64/128/256/512. Every selected-start
comparison is within the prespecified 1e-5 relative-gradient tolerance; the
maximum observed chunk discrepancy is 8.04e-06.
All nine selected-start losses also satisfy the declared 1e-5 absolute tolerance;
the maximum disagreement is 2.95e-08.
This preserves the mathematical full objective while changing FP32 accumulation
partitions. Hardware and rounding can change long continuations; bitwise
agreement with a chunk 64 continuation or the historical trajectory is not claimed.
The maximum source-boundary gradient discrepancy is
6.84e-06.
ResNet's BN groups never change. The prior failed replay qualification at
Run062 anchors 2000/3000 is not repaired by this extension.

Run207 freezes the parallel placement and shared runner for the six later
ResNet windows. It reuses completed 4090 control Pods after verified retrieval,
with two independent windows per worker. Both reach 256 before extensions;
persistent windows then resume from their own model/optimizer/RNG state and
unapplied gradient at 256. Raw baseline/extension evidence and derived phase
configs are retained separately. The largest resume-gradient discrepancy is
4.66e-07, within 1e-5.
The merged outputs preserve all verified prefix rows. Small recomputation
rounding differences do not imply bitwise-identical future trajectories.

Run208 freezes four checkpoint handoffs to shorten the parallel tail: three
later ResNet profiles resume their measured lag 256 states on independent 4090
workers; parent 125/anchor 500 Transformer resumes its measured lag 512 state on
a 5090. Exact scalar prefixes and sparse raw vectors are preserved. Each new
worker must recompute the captured unapplied gradient within the original 1e-5
tolerance before stepping. Old sources stop only after replacement progress
proves qualification. Hardware/rounding can alter subsequent trajectories;
these are qualified continuations, not bitwise historical replays. Source
selection follows the declared handoff activation, not curve outcome.

The new run retains exact per-lag scalar products/norms, full gradients at 11
and the final adjacent pair, and final model/optimizer/RNG recovery states.
The last gradient is measured but not applied. Source references and all source
lag 0..10 vectors remain in Run205. This run does not retain every new full vector
and cannot supply arbitrary future all-pairs or per-parameter lag analyses.

## Execution and cleanup

The cohort used 20 task allocations (7 RTX 4090, 13 RTX 5090),
with at most twelve workers concurrently. Live prices were USD 1.19/GPU-hour
for Secure RTX5090 and USD 0.89 for Secure RTX4090. Some starts failed strict
gradient qualification on 5090 and were retried on the source 4090 hardware;
the tolerance was not relaxed. Thus GPU hardware is not matched across every
LR and anchor; initial numerical agreement does not prove identical future
trajectories. All failed attempts and costs are retained in closeout.
An intentional handoff stop can leave a signal exit code after the scientific
manifest and selected window have completed. Those windows are retained only
with completed window records and verified vectors, checkpoints and file hashes;
the process exit code alone is not the scientific acceptance criterion.
The approved cap was USD 30 and four hours; runtime storage used 30 GB container
disk and 10 GB Pod volume per worker after the API rejected the prepared 8 GB
volume request. No network volume was created. Detached jobs emitted manifests,
progress and events; the supervisor monitored, retrieved, verified and deleted
workers individually. An independent budget/deadline guard bounded compute.
After measured small retention/transfer costs, the operational graceful trigger
was refined from USD 25 to USD 29.5 within the same approved USD 30 envelope, retaining
a USD 29.6 hard compute stop and four-hour deadline. The ten-minute deadline
retrieval margin remained. This changed infrastructure controls only; scientific
inputs and the lag-horizon rule remained frozen.

Median complete gradient plus diagnostic times were
0.013 s MLP,
2.569 s ResNet and
4.178 s Transformer.
Allocation to last deletion took 197.2 minutes; aggregate allocation was
28.341 GPU-hours. Compute is approximately
USD 29.48, provisional pending posted billing/storage.
Task-ID billing reads in billing.json contain only currently posted charges;
hourly buckets may be partially posted. They are not a final settled invoice.
All 674 retrieved files / 2.847 GB passed
size/SHA-256 checks before their Pod was deleted. There were 2
unreachable setup-only workers: available platform logs and monitoring snapshots
were preserved, but full remote setup files could not be retrieved. No scientific
job had started on those workers. All 20 task Pods were terminated;
no task cloud storage remains. Spark and unrelated account resources were
untouched. No manuscript edit or push was made.
