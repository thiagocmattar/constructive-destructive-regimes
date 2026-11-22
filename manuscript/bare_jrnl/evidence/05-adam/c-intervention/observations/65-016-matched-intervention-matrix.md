> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [016-matched-intervention-matrix.md](../../../../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/016-matched-intervention-matrix.md).

# H051: matched four-family intervention comparison

- **Status:** all 36 initialization-0 conditions completed; 11,904 post-fork updates, 11,940 full-gradient boundaries, 180 recorded held-out evaluation states and 108 exact gradient/update audits. Saved terminal states additionally reproduce all 36 training gradients/losses and all 48 full terminal held-out split losses exactly.
- **Question:** at identical starting state and horizon, which controls reduce excursions while preserving learning and how does functional motion change?
- **Comparison:** unchanged Adam plus immediate LR .75/.5/.1; beta2 .95/.9/.5; raw global L2 clipping .5/1; distributed cap q25/q50/q75. The preselected main representation-inspired method remains q50.
- **Matched setup:** MLP 512-800 (5000-image full objective and probe); ResNet 256-576 (50000-image objective; intact 1000-image probe); Transformer 544-928 (2088600-token objective; 64-sequence/2240-position probe). Same full optimizer/RNG/data state and runtime within each model. Native prefixes start at initialization 0.
- **Calibration:** fixed per-module unit-score quantiles over untreated outgoing states f-8 through f-1; no treatment outcome or held-out information enters the budgets.
- **Runtime:** PyTorch 2.11.0; CUDA 13.0/GB10 for MLP and CUDA 12.8/RTX4090 or RTX5090 for ResNet/Transformer. Exact predecessor/fork and eight-update cross-worker identities are verified. These are newly qualified native trajectories, not assumed historical bitwise reproductions.

| Case: q50 versus Adam | PV reduction % | Mean loss change % | Final loss change % | Last-32 change % | Terminal test change % |
|---|---:|---:|---:|---:|---:|
| mlp | +87.75 | -32.18 | -71.27 | -32.56 | +0.53 |
| resnet | +2.77 | -9.42 | +20.02 | +21.47 | +2.42 |
| transformer | +96.23 | -19.84 | -23.31 | -38.97 | +27.52 |

## Main findings

- **Immediate LR:** 0.1x LR removes 99.6%/99.0%/100% of positive loss variation in MLP/ResNet/Transformer, but raises ResNet mean loss by 89.5%. At 0.5x, mean loss falls 25.0%/11.4% in MLP/Transformer and rises 27.2% in ResNet. Smoothness and learning progress are distinct outcomes.
- **Late beta2:** All three switches increase MLP mean loss and positive variation. ResNet is mixed: beta2=.9 lowers mean loss 3.5% while increasing positive variation 63.3%. Transformer mean loss improves 9.6-16.8%, but beta2=.95 increases positive variation 7.1%, and all three switches worsen terminal test loss by 29.3-40.2%.
- **Global raw-gradient clipping:** Both thresholds increase mean loss and positive variation in MLP and ResNet. Transformer clipping at .5 reduces positive variation 87.7% and mean loss 17.8%, while increasing terminal test loss 18.3%. A raw-gradient threshold does not impose a bound on Adam parameter updates or propagated functional movement.
- **Cap quantile sensitivity:** All three quantiles lower mean training loss in all three cases. q50 has the lowest observed mean among the tested settings in MLP and Transformer; ResNet q25/q75 improve mean, positive variation and final/late loss more than q50. Quantile sensitivity is nonmonotone, and q50 terminal test loss increases 0.53%/2.42%/27.52%. No setting wins every metric.
- **Functional motion:** q50 reduces the largest output RMS jump by 83.2%/23.1%/62.9% (MLP/ResNet/Transformer). Total output-motion energy changes by -88.6%/+39.7%/-55.4%, while reversal fractions are 89.2%/66.9%/95.8% of updates, versus 99.3%/36.9%/36.7% under Adam. The central message is smaller extreme functional jumps with continued oscillation; cumulative movement can increase.
- **Exact MLP loss accounting:** For output-space loss rises above 1e-8, all 161 unchanged-Adam rises and all 70 q50 rises have a negative linear term outweighed by positive quadratic movement; LR x0.5 has 11 such rises. The exact identity shows how excessive output displacement can outweigh a corrective linear contribution, even when the latter remains favorable.

- **Connection to V-B:** the new ResNet control exactly reproduces all 187 retained losses over states 264-450. All 732 stage cosine/RMS values in the selected 390-450 tracing window agree within 1e-12. This is the same measured +311.53% excursion at 415. [Fixed-window comparison](../data/matched-tracing_overlap.json).
- **Intervention in the traced window:** The ResNet control reproduces the previously selected 390-450 tracing window. q50 lowers its maximum loss from 0.4407 to 0.1736 (60.6%), while the whole 256-576 continuation has only 2.8% lower positive variation because later oscillations remain. This is a secondary readout in the already selected window; whole-horizon outcomes remain primary. The MLP comparison uses 512-800, after its earlier 451-501 tracing window.

- **Full outcomes:** [36-row CSV](../tables/matched-outcomes.csv), [readable table](../figures/23-matched-intervention-outcomes.pdf), [editable TeX](../tables/matched-outcomes.tex). All settings are retained, including unfavorable outcomes. [Editable component-cost table](../tables/matched-costs.tex); all 36 timing rows remain in the data.
- **Reduction:** primary mean excludes the shared fork; PV sums all positive post-fork loss increments. Final and last-32 losses are distinct. Excursions use the complete untreated prefix plus each treated continuation, the original fixed pre-rise reference, earliest maximum and first return; carried-in and right-censored events remain explicit. Event counts do not create independent replicates.
- **Motion:** stage statistics reduce all fixed-probe coordinates. Squared-motion energy sums coordinate count times squared RMS; reversal energy uses negative consecutive-change cosine. Summaries exclude the common incoming fork change. Auxiliary correct-label log probabilities reproduce probe cross-entropy; MLP output-space loss decomposition is checked separately.
- **Held-out coverage:** full CIFAR-10 test 10000; WikiText-2 validation 217630/test 245550 supervised tokens. Five fixed states per arm; the fork record is shared. There are 147 distinct evaluation states across the three model trajectories (196 full split evaluations), not 180 separately recomputed fork evaluations.
- **Timing:** synchronized gradient, probe capture, optimizer, controller and evaluation components plus remaining wall time. Probe time covers the forward capture and cap inputs; cosine/RMS reductions, other scalar diagnostics, logging and checkpoint I/O lie in remaining wall time. Controller time alone is not the full deployable method overhead. Memory is the process high-water allocation through each arm, an upper bound rather than an isolated per-arm peak. GPU class is fixed within model, but cloud hosts differ. No efficiency advantage is claimed.
- **Uncertainty:** one initialization/model and one prespecified continuation window. No seed confidence intervals, independent-step tests, universal optimizer claim, or selection of a favorable endpoint. Settings have unequal effective strength/tuning grids. The cap bounds direct affine displacement, not total propagated motion or loss.
- **Adversarial interpretation:** post-fork trajectory changes establish controlled intervention effects in these cases; they do not establish that representation reversal is the unique cause of spikes. Retained-moment late beta2 switching also changes native bias correction and is not a test of training from initialization with lower beta2. The study does not compare against AGC, FOOF or Kosson's method.
- **Deferred:** initializations 1/2 and cap x LR x initialization remain separate TODOs with previously recorded ETC/costs; neither is run or needed to report this bounded comparison.
- **Provenance:** [source hashes and reductions](../data/matched-results.json), [independent checks](../data/matched-verification.json), [figure coordinates and sources](../data/matched-figure_manifest.json), [visual review](../data/matched-visual_verification.json). Canonical raw tensors/histories remain in Runs293-295 on owned Spark, with run-level retention inventories.
- **Reproduction:** `60_reduce_matched.py`, `64_verify_matched_reductions.py`, `61_plot_matched.py`, then `62_package_matched.py`; source training worker is `src/eos_tasks/matched_interventions.py`. Raw replay audit: `59_verify_matched.py`.
## Figure captions

### Figure 7: matched intervention training losses

- **Asset:** [17-matched-intervention-loss.pdf](../figures/17-matched-intervention-loss.pdf).
- **Caption:** Matched full-objective Adam continuations for CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer (columns). Rows change only the immediate learning rate, beta2, global raw-gradient clipping, or distributed affine-displacement cap. Gray is the same unchanged Adam control in every row of a column. All requested settings and every training state are shown without smoothing; loss axes are logarithmic and identical within each model. Each treatment starts from the complete common state at f=512/256/544 and stops at e=800/576/928. The dashed line marks the fork. One initialization per model; settings do not imply equal effective strength.

### Figure 8: spike burden versus learning progress

- **Asset:** [18-matched-spike-learning-tradeoff.pdf](../figures/18-matched-spike-learning-tradeoff.pdf).
- **Caption:** Every intervention setting is compared with the common unchanged Adam baseline at the origin. Horizontal position is the percentage change in mean training loss over f+1 through e; vertical position is 100 times one minus the treatment-to-baseline ratio of positive loss variation, PV=sum max(L_s-L_(s-1),0). Leftward and upward changes are favorable. Family shapes/colors and setting labels retain the complete grid; large negative vertical values denote increased excursions. Axes are model-specific. The observations are paired trajectories at one initialization, not independent estimates with seed-level confidence intervals.

### Figure 9: matched q50 loss and functional motion

- **Asset:** [19-matched-q50-stage-heatmaps.pdf](../figures/19-matched-q50-stage-heatmaps.pdf).
- **Caption:** Unchanged Adam versus the preselected distributed q50 affine-displacement cap at the common forks. Rows show full training loss, baseline/cap consecutive-change cosine, and baseline/cap displacement RMS at 5/6/6 measured stage endpoints. Changes are placed at their incoming endpoint s and cover f+1 through e. Both RMS maps use each stage's same maximum under unchanged Adam in that post-fork window; they are not separately normalized or divided by activation magnitude. The cosine scale is [-1,1]; the shared RMS scale includes all observed values. Dashed markers locate the peak of the baseline's largest relative loss excursion in the fixed window, using the same pre-rise reference and first-return rule as the outcome table. They are descriptive reference markers, not intervention triggers. MLP uses all 5000 examples; ResNet uses an intact 1000-image normalization group; Transformer uses 64 sequences/2240 positions. Outputs are raw MLP coordinates or cross-entropy log probabilities. No smoothing, SVD refit, or whole-network displacement bound is implied.

### Supporting stage motion for every setting

- **Asset:** [20-matched-all-setting-stage-heatmaps.pdf](../figures/20-matched-all-setting-stage-heatmaps.pdf).
- **Caption:** Eleven pages retain the complete intervention grid, in the same LR, beta2, clipping and cap order as the loss matrix. Each page uses the Figure 9 layout, full fixed windows, the unchanged baseline, and the same baseline stage RMS denominators. RMS color limits are shared across the three models within each page and may differ between pages to include every value. Page headers identify the treatment. These maps test functional motion under all controls; they do not identify a unique causal source of instability.

### Full held-out test trajectories

- **Asset:** [21-matched-heldout-loss.pdf](../figures/21-matched-heldout-loss.pdf).
- **Caption:** Complete held-out test loss at f, f+32, midpoint, e-32 and e, with the same family/setting order as Figure 7. The fork evaluation is shared; every later evaluation is complete, with 10000 CIFAR-10 images or 245550 WikiText-2 test tokens. The Transformer's additional 217630-token validation results are retained in the accompanying data. Lines join only the five prespecified observations; no held-out checkpoint, setting or window selection is performed. Training improvements must be interpreted together with these held-out outcomes.

### Applied movement and controller activity

- **Asset:** [22-matched-update-and-controller-activity.pdf](../figures/22-matched-update-and-controller-activity.pdf).
- **Caption:** Page 1 shows the actual stored parameter-increment norm for every treatment and the unchanged baseline, on logarithmic axes shared within each model. Page 2 shows the retained raw-gradient factor for clipping and the fraction of targeted units capped at each update for q25/q50/q75. Values are placed at the completed-update endpoint. Clipping acts before Adam; caps act on the native stored proposal while retaining its updated moments and clock. Actual trajectories diverge after treatment, so parameter norms alone are not matched functional strength. Component timings and memory are reported separately, without a cross-host method-speed ranking.

### All 36 matched intervention outcomes

- **Asset:** [23-matched-intervention-outcomes.pdf](../figures/23-matched-intervention-outcomes.pdf).
- **Caption:** One model per page, all 12 conditions, identical windows and full test coverage. PV reduction and mean-loss change are relative to unchanged Adam; absolute final and final-32-state mean losses preserve endpoint context. Green highlights the preselected q50 method rather than an outcome-selected winner. The accompanying CSV includes all event, activity, validation, endpoint and relative-effect columns. Undefined effects use N/A; there are no seed error bars from this initialization-0 study.
