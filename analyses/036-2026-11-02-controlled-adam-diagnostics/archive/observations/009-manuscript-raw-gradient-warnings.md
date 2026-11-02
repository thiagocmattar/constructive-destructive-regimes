# Observation009: manuscript views of raw-gradient warnings under Adam

## Question, source and scope

The author replaced the old Adam exposition/Pythia illustrations with the
completed Analysis036 experiments. The central idea is: **In Adam,
raw-gradient dynamics provide information about impending loss excursions
beyond recent loss history, even though Adam transforms those gradients
before updating the parameters.** The evidence here consists of the nine
controlled full-batch Run215 arms, one frozen initialization per case, and the
existing exploratory forward-time analysis. The formulation states the idea
generally; the methods specify the empirical coverage. This revision adds no
training, metrics, seed, hypothesis, run, cloud allocation or spending.

Sources are `full_diagnostics_manifest.json`, `full_unsmoothed_manifest.json`,
`peak_micro_three_models_manifest.json`, `peak_temporal_results.json`, and the
hash-pinned original Run215 metrics/boundaries they reference. Observation005
records the full cohort; Observation008 records all episode/forecast definitions,
timing, false alarms, negative controls and interpretation limits. The new
revision manifest pins their hashes and the raw inputs again.

## Figures, reductions and captions

`figures/adam_manuscript_overview.pdf` is a six-row, three-column view of all nine
arms: training loss, raw projected gradient change, raw cosine, stored-momentum
cosine, raw-gradient norm and stored-momentum norm. Loss is raw; all other rows
retain the complete 50-observation trailing arithmetic means from the verified
full figure. There is no padding, thinning, log-space averaging or change to
the measurements. The applied-update clock of the older full figure shifts
left by one for vector rows: horizontal state s means theta_s, where the
manuscript's g_(s+1) and m_(s+1) are available before update s+1. Loss keeps its
state index. This is the same availability clock as the temporal analysis.

Caption: Controlled Adam training at three constant rates per case. Training
loss is unsmoothed; other rows use complete 50-observation trailing means.
All rows align to parameter state theta_s, with g_(s+1), m_(s+1) available before
the next update. Momentum denotes the stored, uncorrected first moment. The
reference at two is not an identified Adam stability boundary. One
initialization per case.

`figures/adam_manuscript_spikes.pdf` is a three-row, three-column view of the
same three existing selected episodes: loss, raw projected change, and raw
gradient/momentum norms divided by their respective first value in each crop.
All values are unsmoothed; normalization only expresses proportional growth.
Each highest-rate arm contributes its completed episode with the largest
loss peak/baseline ratio, selected from loss alone. Peaks 703/422/794 and crops
693--729/412--481/784--852 remain unchanged. A dotted vertical marks the
episode's 10% onset; a dashed vertical marks the peak. The plotting reference
loss and normalization baseline are the crop start, distinct from the
episode's preceding-16-state minimum. Every point in each crop is retained.

Caption: Unsmoothed views of the completed episode with the largest
peak/baseline ratio at each case's highest Adam rate. Each crop begins ten
states before its peak and ends at the first subsequent loss below its starting
value. Dotted verticals mark 10% episode onset, dashed verticals the peak. Norms
are divided by their own starting value. All diagnostics use the parameter-state
availability clock. Quantitative analysis includes repeated events and
non-spike states.

These two PDFs contain 66 plotted series and 53,777 values. They are copied
byte-for-byte into the manuscript's active `figures/analysis036/` directory.
Earlier eight-/nine-row figures are preserved. The active manuscript now has
seven compiled figures; the five earlier GD/SGD PDFs are unchanged.

## Forecast table, interpretation and uncertainty

`manuscript/draft/tables/analysis036_forecast.tex` presents positive prevalence
and average precision of the loss-history and added-projected-change models
for each case and pooled predictions. Its 12 scalars are rounded from the
existing forward-test results; no refitting or test selection occurs. Per-case
rows subset the pooled models' predictions, rather than fit new task models.

Pooled AP rises 0.260 to 0.454 beyond task/rate, training age and recent loss,
and both AP and Brier improve in all five forward blocks. This supports the
central incremental-information claim within the measured coverage. The main
text also retains the 13/17, 3/3, 7/7 highest-rate large-episode warning counts,
187/891 true-positive threshold states, and no-episode low-rate negative
controls. Warning coverage and alarm precision have different denominators.

Projected change above two already requires negative raw cosine and raw-norm
growth. Those are not independent corroborations. The ordering around the
Transformer peak precedes the pronounced rise, not its very first small loss
increase. A higher loss need not imply a larger gradient norm, and temporal
ordering does not establish a causal feedback mechanism or an effective
scheduler. The continuous forecast comparison is the primary evidence;
threshold crossings and selected examples explain it rather than replace it.

There are no initialization-level confidence bands. Repeated states and
episodes within one trajectory are dependent. The originally inspected peak
motivated the exploratory analysis; later blocks are held out from fitting,
not pristine confirmatory evidence for a later selected rule. Initial-gradient
qualification does not isolate all subsequent cross-hardware effects. No
generalization, Hessian, universal stability boundary or intervention benefit
is established. These limitations are recorded in the manuscript methods.

## Generation and verification

The generator is
`manuscript/draft/revisions/2026-11-02-adam-feedback/01_prepare_figures.py`.
Its local `figure_manifest.json` records all arrays, transformations, table
values, source hashes and output hashes. The companion `02_build_and_verify.py`
independently reconstructs every plotted value from raw Run215 histories,
using math.fsum for window means. It rechecks pointwise algebra, exact state
clocks, table rounding and all displayed scientific examples. The existing
independent temporal verifier rebuilds all 4,534 forecast labels, episode
definitions, early-warning leads and scores. Earlier GD/SGD/table/math checks
remain part of the new manuscript build. Numerical checks pass; final visual
acceptance is recorded in the revision's `review.json`.
