> Evidence snapshot; original scientific content retained. Relative links were rebased.
> Source: [OBSERVATIONS.md](../work/figure_revision_20261114/OBSERVATIONS.md).

# Consolidation figure revision, 14 November 2026

- **Purpose:** implement the author's presentation feedback for IV and V-A/B. Seven publication PDFs; existing measurements, no training or new captures.
- **Generator:** [01_figures.py](../work/figure_revision_20261114/01_figures.py). [Numerical/source manifest](figure_revision_manifest.json); [independent verification](figure_revision_verification.json); [editable captions](figure_captions.tex).
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

- **Sources:** Analysis063's MLP 450--500, ResNet 280--310 and Transformer 552--600 outgoing rows; all values and fitted windows are preserved. Displayed completed steps become **451--501 / 281--311 / 553--601**. The final MLP loss is read from the same retained full-window continuation and all overlapping losses are checked for equality.
- **Stage reduction:** select measured architectural endpoints, without averaging layer cosines. MLP: both linear/tanh pairs and output. ResNet: first convolution, final output of each of three residual stages, pooling, output. Transformer: embedding, attention and final normalized output of each block, output. These are the corresponding stages used in the q50 presentation, rather than every internal operation.
- **Outputs:** raw linear MLP output; all-class/all-vocabulary log probabilities for ResNet/Transformer. Heatmap cosine retains [-1,1]. Each displacement RMS row is divided by its own maximum over the fixed window. This shows time variation within a stage, not absolute amplitude comparisons between stages/models.
- **Probe coverage:** MLP all 5,000 training images; ResNet first intact original 1,000-image BatchNorm group out of 50,000 images; Transformer first 64 sequences/all 2,240 supervised positions out of 2,088,600 training tokens. Full and probe losses are shown separately for ResNet/Transformer; they coincide for MLP. Retain original model modes, fixed inputs and stochastic controls. One initialization/probe/window per architecture.
- **Dashed markers:** blue = first recorded negative **output-change** cosine in the window. Black = first full-objective loss at least 10% above the pre-peak minimum. Define the peak as the largest full-objective loss in the displayed window and the baseline as the minimum preceding it. This is a retrospective event annotation, chosen using loss alone; heatmaps are unsmoothed. Both markers use identical completed-step coordinates on all three rows.

| Model | Output reversal first observed | Baseline | First increase from baseline | First loss >=1.10 baseline | Peak |
|---|---:|---:|---:|---:|---:|
| MLP | <=451 (already negative at entry) | 465 | 466 | 470 | 474 |
| ResNet | <=281 (already negative at entry) | 286 | 287 | 289 | 293 |
| Transformer | 556 (observed zero crossing) | 557 | 558 | 561 | 562 |

- **Interpretation:** Transformer output reversal is observed two completed steps before the selected loss begins rising and five before the +10% crossing. MLP/ResNet reversals are already present at capture entry: their onset is left-censored, so 451/281 are not onset estimates. These selected windows support temporal characterization, not a validated warning system or a unique causal origin. The +10% marker must not be described as the first loss increase.
- **Caption:** Loss and functional motion across three Adam trajectories. Columns show MLP, ResNet and Transformer; rows show full/probe loss, consecutive representation-change cosine and displacement coordinate RMS divided by each stage's own window maximum. Rows within heatmaps are measured stage endpoints, ordered through the network. Each change A_s-A_(s-1) is placed at completed step s, alongside the loss at that state; no time smoothing is applied. Blue dashed lines mark the first observed negative output-change cosine; reversal is already present at window entry for MLP and ResNet, so its onset is unobserved there. Black dashed lines mark the first full loss at least 10% above the minimum preceding the largest displayed peak. Both markers are repeated through loss and heatmaps. The fixed probes contain 5,000 images, 1,000 images and 2,240 token positions, respectively. MLP probe and full loss coincide; the other probes are subsets. Normalized RMS colors compare time within a stage, not absolute magnitudes across stages. These are descriptive selected windows from one initialization each.

## Companion: temporal structure and SVD

- **Sources/reduction:** the exact same original window fits as Figure 6: 51/31/49 displacement rows, without centering, normalization, resampling or refitting. Spectra, stage energy/overlap and signed coefficients are copied from the verified reductions.
- **Caption:** Temporal structure of the representation-change windows in Figure 6. Top: cumulative singular-value energy for output changes and raw gradients; the dotted reference is 95%. Middle: first-mode energy, first-ten-mode energy and top-ten temporal subspace overlap with output, at the selected architectural stages. Bottom: the leading output coefficient divided by the square root of its coordinate count. Each coefficient uses one fixed basis and sign convention over the original window. All time coordinates place a change at the completed step where it is observed. The output changes use raw MLP outputs and ResNet/Transformer log probabilities; raw gradients are not differenced. Temporal overlap compares time-domain subspaces, not identical neuron directions. Different probes and window lengths preclude a controlled intrinsic-dimension comparison. SVD concentration alone is time-permutation invariant and does not establish a causal source or spike specificity.

## Verification and limits

- Generator checks source hashes, numerical arrays, exact stage selection, shifted coordinates, common smoothing support, marker crossings, heatmap matrices and figure boundaries.
- The independent checker reopens every PDF, verifies embedded fonts/text/page bounds, compares unchanged Y arrays to the prior figure manifests, and checks the event clocks directly from retained boundary data.
- All seven Poppler renders are inspected at publication scale. The current manuscript and intervention package must remain byte-identical to the pre-revision versions.
- No new independent sample, run, hypothesis, confidence interval or causal experiment is created by this presentation revision.
