# Analysis005: phase-dependent temporal raw-gradient alignment

Latest: [1-400 overlay with log-loss differences](figures/training_size_overlay_log_loss_diff_n5_1_400.pdf).
The loss panel uses an N=5 mean of ln(L_t/L_(t-1)), approximately relative loss
change per update; other panels remain raw. See
[observation007](observations/007-log-loss-difference.md).

[1-400 overlay with short loss-difference mean](figures/training_size_overlay_loss_diff_n5_1_400.pdf).
The loss panel uses mean(L_t-L_(t-1)) over N=5 updates; negative means falling
loss. The other three panels remain unsmoothed. See
[observation006](observations/006-short-loss-difference.md).

[Separate unsmoothed 2x2 overlay, updates 1-400](figures/training_size_overlay_raw_1_400.pdf).
All four variables use raw values and solid coloured curves; see
[observation005](observations/005-unsmoothed-first400.md) for coverage and checks.

Full horizon: [smoothed 2x2 model-size overlay](figures/training_size_overlay.pdf).
All panels overlay 14M/31M/70M using solid lines in three colours. Loss is
unsmoothed; other variables use complete 20-update trailing means.
Update 712 is excluded before smoothing. [Observation004](observations/004-smoothed-size-overlay.md)
compares early loss changes and norms: larger first-10 loss drops coincide with
larger norms, but norm ordering is not monotonic over the first 50 updates.

October 17 follow-up: [training overview](figures/training_overview.pdf), with
three model-size columns and four rows: loss, eta*kappa_hat proxy, rho and raw
gradient norms. All 712 updates per size are retained; no new training is needed.
Initial, midpoint and final recovery checkpoints are present locally. See
[observation003](observations/003-training-overview.md) for paths, definitions,
checks and the AdamW/minibatch interpretation limit. The six original PDFs remain.

Status: complete. All three full histories are retrieved and verified; all six
task-owned workers, including infrastructure failures, are deleted. Six PDFs
passed rendering and visual inspection. Multi-seed validation remains deferred.

[Read the conclusion](REPORT.md), [phase lag profiles](figures/lag_profiles.pdf),
and [all-lag cumulative interaction](figures/cumulative_lag_interaction_all_lags.pdf).
The [observations](observations/INDEX.md) record all six captions and reductions.

Main result: negative adjacent raw-gradient interaction coexists with positive
exact all-pair interaction in all three middle windows. Longer-lag cosine retains
a positive background. The early/late smooth-decay-to-zero expectation is not
uniformly supported through lag64. These are descriptive one-seed measurement
replays; their numerical trajectories differ from the earlier reference runs.
Question3 of manuscript/ideas/reflections.md motivates measuring beyond lag1.
User expects early smooth decay toward zero, middle oscillation during local
cancellation, and late decay toward zero. These are exploratory expectations,
not imposed fits or guaranteed properties. A curve may retain a positive plateau.

Runs017/018/019 replay14M/31M/70M,seed1235,all712 original-recipe updates.
Reused: immutable architecture/initial parameter SHA,model/data seeds,pinned
MiniPile/tokenizer cache bytes,ordering,batch/accumulation,AdamW/schedule/precision.
Missing raw history cannot be recovered from sparse checkpoints. New diagnostics
record four independent fixed CountSketch vectors every accepted optimizer update;
16384 buckets per map,FP32 history andFP64 offline reductions. Exact original norms
normalize projected dot products. The map is fixed across time and training RNG
is isolated. This is an empirically calibrated sketch,not a worst-case JL guarantee.
The shared scientific helper is in src because all three new runs require it;
the executed training source and original artifacts remain unchanged.

Windows fixed before measurement:early1-128,middle293-420,late585-712.
Primary: signed cosine at every integer lag1-64,using only within-window pairs.
Secondary: all127 lags with decreasing pair coverage,window Gram heatmaps,
norm-weighted interactions,shuffled-order reference,and exact all-lag aggregate.
No assumption of stationarity,zero long-lag mean,or universally constructive early/
destructive middle/constructive late windows. Report lag1 and negative frequency
within each actual window. Shared update windows are primary across sizes.

Projection error:exact lag1 every update,randomly anchored lag2/4/8/16/32/64;
four map profiles show projection sensitivity,not training-seed uncertainty.
Compare larger-lag exact pairs separately by window and sign margin; aggregate
closure alone does not validate individual lag curves. Calibration thresholds:
ensemble pair cosine RMSE<.02 and maximum absolute error<.06. Small features
below demonstrated error remain unresolved; no claims from projection sign alone.
All final partial-batch evidence stays recorded; primary late window includes712,
with an explicit tail-exclusion sensitivity in the analysis.

Completed workers: Secure RTX4090 at the live allocation price of USD0.74/GPU-hour.
Initial Community5090 attempts had SSH/host-driver failures before training;
three failed owned workers were removed. Runs018/019 use unchanged scientific
inputs in attempt-02; Run017 uses scientific attempt-01 on its replacement host.
Three parallel workers,at most4hours each,USD15 aggregate cap including retries
and storage. Actual cloud elapsed time including retries/retrieval:3h12m;
estimated total USD6.54,provisional. See `closeout.json` and `billing_final_snapshot.json`.
60GB container+40GB persistent pod storage each;running storage
$0.10/GB/month for both types per live Runpod documentation. No network volume.
Initial qualified ETC2-2.5hours elapsed,based on this project's measured69-94minute
full passes,refreshed after GPU smoke and during training. Detached jobs,persistent
events/recovery,local deadline stop guards,20minute retrieval margin. Verify sizes/
SHA256 and terminal scientific checks passed before deleting workers. Local raw
sketches/checkpoints and archives are retained;task cloud storage is zero. All
deadline guards are stopped. Unrelated resources received no actions.

Scientific scope:raw temporal gradient coherence and cancellation. No inference
of stability,generalization,causal benefit,or actual AdamW displacement. Single
training seed for first test;multiple projection seeds are not independent runs.

Reproduction: `03_analyze.py` reduces accepted artifacts to `results.json`,
`curve_data.csv` and the six PDFs; `08_record.py` creates the observations;
`artifact_validation.json` records hashes, checks and completed visual QA.
`04_accept.py` independently verifies retrieval/scientific evidence;
`09_close_worker.py` and `11_closeout.py` record attributed resource cleanup.
Executed scientific inputs and supplied manuscript assets remain unchanged.
