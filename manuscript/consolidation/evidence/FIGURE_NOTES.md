> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [OBSERVATIONS.md](../work/resnet_peak_revision_20261114/OBSERVATIONS.md).

# Consolidation figure revision, 14 November 2026

- **Purpose:** implement the author's presentation feedback for IV and V-A/B. Seven publication PDFs; existing measurements, no training or new captures.
- **Current Figure 6:** [generator](../work/resnet_peak_revision_20261114/01_figure.py), [manifest](05-adam/b-tracing/data/resnet_peak_figure_manifest.json), [verification](05-adam/b-tracing/data/resnet_peak_figure_verification.json). Other views retain the [first revision generator](../work/figure_revision_20261114/01_figures.py) and [checks](figure_revision_verification.json). [Current editable captions](figure_captions.tex).
- **Selected views:** four GD/SGD figures; compact high-rate Adam overview; one three-model loss/motion figure; one separate temporal-structure figure. The previous three detailed tracing PDFs remain in the package's `background/figures/` and their canonical analysis folders.
- **Scope:** current manuscript, human inputs, raw evidence, historical generators and intervention figures are preserved.

## Common progress coordinate

- **Label:** `Training step s`. Here s is the number of completed optimizer updates; s=0 is initialization. One step is one complete optimizer update, not a batch fragment, epoch or forward pass.
- **Loss:** L(theta_s). **Raw gradient:** g_s evaluated at theta_s, before update s to s+1. Its norm, consecutive cosine and Q_s=eta*kappa_hat_s are placed at s, after that gradient is computed.
- **Legacy GD conversion:** the old gradient index t evaluates g_t at theta_(t-1), so norm/cosine/Q coordinates move from t to s=t-1. Spectral estimates retain their actual measurement state s; the previous artificial +2 display offset is removed. R_s retains the prefix ending at theta_s, without a shift.
- **Representation conversion:** the retained outgoing row t describes A_(t+1)-A_t. Plot it at s=t+1, where that change is observed, alongside L(theta_s). Thus the displayed cosine/RMS is the existing rho_(s-1)^ell / a_(s-1)^ell in the consolidated outgoing equations. This is an explicit coordinate conversion, not a redefinition or recomputation of the source diagnostic.
- **Other axes:** backward lag k counts steps back from a fixed endpoint; batch size and number of modes retain their own units. They are not training-progress axes.
- **SVD:** fit rows are unchanged. Only coefficient display coordinates move by +1. No cropped basis, refit, pooling, or sample averaging is introduced.

## Figure 1: loss and smoothed GD curvature

- **Sources:** Analysis024's authenticated single-row/spectral arrays and nine original metric histories. Sources and hashes are recorded in the manifest.
- **Reduction:** loss remains raw. Both curvature series use the **same centered 101-step arithmetic mean** on the common state grid 1--2999. At the ends, both use the same truncated support. The sparse spectrum, measured at states 0,100,...,3000, is linearly interpolated on this grid before smoothing; no extrapolation is used. The proxy is reconstructed from original gradient norms/cosines and checked against the previous 50-point averages.
- **Reason:** the former trailing proxy average and unsmoothed spectrum had different display filters. Equal centered filtering removes that display-phase asymmetry. It does not force equal crossing times: genuine differences remain, and these are different curvature quantities.
- **Limits:** this is retrospective visual smoothing, including later observations within a centered window. Interpolation does not create new Hessian measurements. Do not infer exact onset or a forecast from these curves. The terminal spectral measurement at 3000 contributes to interpolation but no Q_3000 is invented.
- **Caption:** Training loss and curvature diagnostics under full-batch GD. Columns identify dataset/model and colors identify learning rate. Top: unsmoothed full-objective loss on logarithmic axes. Bottom: eta*kappa_hat_s (solid) and eta*lambda_max (dashed), both smoothed with the same centered 101-step mean; the dotted line marks two. Spectral estimates are available every 100 steps and are linearly interpolated before smoothing. Edge windows are truncated equally. The horizontal axis counts completed optimizer updates; the raw-gradient diagnostic at s uses gradients at s-1 and s, while the spectrum is indexed by its measurement state. Spectral objectives use all 5,000 MLP images, 5,000/50,000 ResNet images and 2,500/59,680 Transformer sequences. One initialization per model; smoothing is for display and does not establish exact transition timing.

## Figure 2: GD reversal, norm and cumulative displacement

- **Sources/reduction:** unchanged Analysis024 cosine/norm values, with complete trailing 50-observation means; unchanged unsmoothed prefix R. Only gradient time coordinates move by -1 and notation becomes s.
- **Caption:** Consecutive gradient cosine, gradient norm and normalized displacement for the GD conditions in Figure 1. Cosine and norm use complete trailing 50-observation means, with norms on logarithmic axes; R_s is unsmoothed and uses all updates completed at step s. Raw-gradient statistics are indexed by the state where the gradient is evaluated. Colors and model ordering match Figure 1. One initialization per model; the curves describe directional interaction and accumulation, not generalization or a stability guarantee.

## Figure 3: fixed-endpoint backward lag

- **Source/reduction:** Analysis035, endpoint theta_2256, GD continuations from state 2000. All numerical coordinates and cosine values are unchanged; gradient subscripts follow the state-indexed convention.
- **Caption:** Backward gradient alignment at the common completed step s=2256. Each point is cos(g_s,g_(s-k)) for one fixed endpoint; top panels show lags 1--256, bottom panels lags 1--32. Learning rates are 0.04, 0.1 and 0.2 for MLP, ResNet and Transformer. These separately recomputed GD continuations use one initialization, with no smoothing or averaging over endpoints. The horizontal coordinate is backward separation in steps, not training progress.

## Figure 4: minibatch comparison

- **Source/reduction:** verified 2 November manuscript figure manifest, with the original Analysis027/037 reductions. All six mean curves and initialization ranges are unchanged and checked against the three initialization means.
- **Caption:** Mean sampled-gradient cosine over 199 adjacent pairs in 200-update continuations, started at initialization or after 2,000 GD updates. Lines first average repetitions within initialization and then weight the three initialization blocks equally; shading spans their range. Hollow rightmost markers denote full GD. ResNet batches comprise complete fixed 1,000-image normalization groups; Transformer batches count prepared sequences. Batch axes are logarithmic. Equal update counts do not imply equal data exposure.

## Figure 5: compact high-rate Adam overview

- **Sources/reduction:** Analysis055's authenticated Run215 histories and unchanged loss-event table, previously copied through Analysis058. All 12 numerical curves and event intervals are unchanged; there is no smoothing or new event selection.
- **Display:** height decreases from 5.65 to 4.65 inches at the same 7.16-inch width. Shading descriptions move into the caption. Q axes start at zero and use only the displayed high-rate arm to set the upper limits (3.5,3.5,4.0).
- **Out-of-view values:** 13 MLP, 17 ResNet and 1 Transformer Q values are below zero; minima are -0.359825, -0.058081 and -0.020606. Values are retained unmodified, not floored at zero. The zero-baseline viewport is an explicit display choice.
- **Caption:** High-rate full-batch Adam trajectories on CIFAR-10/MLP (eta=0.001), CIFAR-10/ResNet-32 (0.003) and WikiText-2/Transformer (0.003). Rows show training loss, projected raw-gradient change eta*kappa_hat_s, consecutive raw-gradient cosine and gradient norm relative to its initial value. Filled gray bands mark pre-rise baseline-to-peak intervals of recovered excursions whose peak exceeds baseline loss by more than 10%; outlined bands mark such excursions with recovery unobserved within the trajectory. Loss and relative norm use logarithmic axes. Projected-change axes start at zero, leaving 13/17/1 negative observations outside the view; data are not clipped numerically. Loss covers steps 0--1024, norms 0--1023 and paired diagnostics 1--1023. One initialization per model; under Adam, projected gradient change is not a validated Hessian estimate or universal stability threshold.

## Figure 6: combined loss and representation motion

- **Current source:** [larger ResNet event](05-adam/b-tracing/observations/63-007-resnet-major-peak.md), re-reduced from the unchanged Run273 control. MLP/Transformer retain the exact original arrays. Displayed completed steps are **451--501 / 390--450 / 553--601**. The original ResNet 281--311 view remains in the first figure-revision folder.
- **Selection:** the last recovered >100% loss excursion before step 512 in the native control. Baseline **0.1070888 at 404**, peak **0.4407024 at 415 (+311.53%)**, recovery **439**. Selection uses loss only; the window includes the pre-rise and recovery. The former peak at 293 rose 31.93%.
- **Stage reduction:** unchanged 5/6/6 measured endpoints, without averaging layer cosines. ResNet uses the first convolution, the outputs of residual stages 1/2/3, pooling and all-class log probabilities. Source boundary s already contains the incoming change A_s-A_(s-1); no additional +1 shift is applied to that source.
- **Outputs and scale:** raw MLP outputs; all-class/all-vocabulary log probabilities for ResNet/Transformer. Cosine spans [-1,1]. Each RMS row divides by its own maximum within the displayed window, comparing time within a stage rather than absolute amplitudes between stages.
- **Probe coverage:** all 5,000 MLP images; the first intact original 1,000-image ResNet BatchNorm group out of 50,000; the original 64 Transformer sequences/all 2,240 supervised positions out of 2,088,600 tokens. Full/probe loss curves remain separate for ResNet/Transformer. Model modes, fixed inputs, stochastic context and full training objectives are unchanged.
- **Dashed markers:** blue marks the first recorded negative output-change cosine in the display; black marks the first full loss at least 10% above the pre-peak minimum. Markers have identical completed-step coordinates across rows. Neither is an established alarm or causal onset.

| Model | First displayed negative output cosine | Baseline | First increase from baseline | First loss >=1.10 baseline | Peak | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| MLP | <=451 (negative at entry) | 465 | 466 | 470 | 474 | See original trace |
| ResNet | 391 | 404 | 405 | 406 | 415 | 439 |
| Transformer | 556 | 557 | 558 | 561 | 562 | 596 |

- **ResNet interpretation:** the first displayed negative value at 391 is an earlier reversal, not a unique onset of the later major excursion. First-convolution movement peaks at 415; the other five measured endpoints peak at 416. Alignment-series correlations with output are .764/.828/.905/.995 for first convolution/stages 1/2/3. These dependent observations establish participation and timing, not causal transmission.
- **Trajectory qualification:** 313 loss states exactly match the native Run223 control. The prior Run243 trace agrees on 48 loss states and 372 stage measurements (maximum difference 6.67e-16). This is the same qualified continuation as the old tracing view, not bitwise replay of Run215's original overview. No new training or SVD is performed.
- **Caption:** Loss and functional motion across three unchanged-Adam continuations. Columns show MLP, ResNet and Transformer; rows show full/probe loss, consecutive representation-change cosine and displacement RMS divided by each stage's own window maximum. Each change A_s-A_(s-1) is placed at completed step s, alongside the loss at that state, without smoothing. Displayed steps are 451--501, 390--450 and 553--601. The ResNet window contains the +311.53% excursion peaking at 415 and recovering at 439. Blue lines mark the first recorded negative output-change cosine in the display; MLP is already negative at entry. Black lines mark the first full loss at least 10% above the minimum preceding the largest displayed peak. Fixed probes contain 5,000 images, 1,000 images and 2,240 token positions. RMS colors compare time within a stage. These are selected single-initialization windows; the separate SVD companion retains the original ResNet window 281--311.

## Companion: temporal structure and SVD

- **Sources/reduction:** the preserved original fits at completed steps **451--501 / 281--311 / 553--601**: 51/31/49 displacement rows (the ResNet fit is distinct from the revised Figure 6 window 390--450), without centering, normalization, resampling or refitting. Spectra, stage energy/overlap and signed coefficients are copied from the verified reductions.
- **Caption:** Temporal structure of the original representation-change windows: completed steps 451--501, 281--311 and 553--601. ResNet retains its earlier small-excursion fit; it does not characterize Figure 6's new 390--450 window. Top: cumulative singular-value energy for output changes and raw gradients; the dotted reference is 95%. Middle: first-mode energy, first-ten-mode energy and top-ten temporal subspace overlap with output, at the selected architectural stages. Bottom: the leading output coefficient divided by the square root of its coordinate count. Each coefficient uses one fixed basis and sign convention over the original window. All time coordinates place a change at the completed step where it is observed. The output changes use raw MLP outputs and ResNet/Transformer log probabilities; raw gradients are not differenced. Temporal overlap compares time-domain subspaces, not identical neuron directions. Different probes and window lengths preclude a controlled intrinsic-dimension comparison. SVD concentration alone is time-permutation invariant and does not establish a causal source or spike specificity.

## Verification and limits

- The [first verification](figure_revision_verification.json) remains the record for the initial seven views, preserved in their original work folder.
- The [current Figure 6 verification](05-adam/b-tracing/data/resnet_peak_figure_verification.json) checks raw boundary indexing, all plotted values, marker crossings, source hashes, embedded fonts and page bounds. MLP/Transformer panel pixels and numerical arrays remain identical.
- The revised Poppler render is visually reviewed. All other PDFs, all intervention files and the intervention PLAN section remain byte-identical; manuscript/human input preservation is also checked by the package verifier.
- No new training, independent sample, run, hypothesis, confidence interval, SVD or causal experiment is created by this reduction.
