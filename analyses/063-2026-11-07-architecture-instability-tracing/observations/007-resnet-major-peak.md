# ResNet tracing around the larger pre-512 excursion

- **Request:** replace Figure 6's small ResNet peak with a larger loss excursion before step 512. Re-run the reduction and figure using existing every-step measurements.
- **Result:** completed steps **390--450**, baseline **0.1070888 at 404**, peak **0.4407024 at 415**, first recovery **439**. The peak is **4.1153 times its pre-rise loss (+311.53%)**, versus **+31.93%** at the previous peak 293. Absolute peak height is slightly lower; severity here means the rise relative to its own pre-rise loss.
- **Selection:** apply the established fixed-level, loss-only excursion detector to the native control; choose the last recovered excursion exceeding 100% before step 512. The 390--450 display includes 14 steps before the baseline and 11 after recovery. No layer diagnostic enters event/window selection. This is retrospective selection, not an unbiased population sample.

## Sources and comparison contract

- **Reduction:** [11_resnet_major_peak.py](../11_resnet_major_peak.py); [all plotted measurements and source hashes](../resnet_390_450_motion.json).
- **Raw source:** [Run273 unchanged-Adam control](../../../runs/273-2026-11-07-resnet-distributed-displacement/README.md), `artifacts/resnet-a/resnet-distributed/control/boundaries.jsonl`. Only its control arm is used; no capped trajectory enters the figure.
- **Matched factors:** same saved Run223 seed-0 prefix, ResNet-32/CIFAR-10, full 50,000-image mean cross-entropy, Adam LR .003, betas (.9,.999), epsilon 1e-8, no clipping/decay, native FP32, FP64 diagnostic reductions, TF32 off. Same fixed first 1,000-image original BatchNorm group, training-mode function and input order.
- **Changed factor:** display window. Six measured endpoints retain every probe coordinate: first convolution; outputs of residual stages 1/2/3; pooling; all-class log probabilities. Stage endpoints agree with the earlier detailed trace; they are not averages of layer cosines.
- **Clock:** at completed step s, use loss L(theta_s) and incoming change A_s-A_(s-1), compared with A_(s-1)-A_(s-2). Read these directly from boundary s. The older transfer summary stores representation diagnostics at outgoing row s-1; using its unshifted row would misalign loss and motion.
- **Reduction:** no smoothing, centering, sample averaging or new SVD. Cosine uses [-1,1]. Each stage's RMS divides by its own maximum over 390--450; colors compare time within a stage, not absolute magnitudes between stages.

## Findings in the revised window

| Stage | Negative / defined change cosines | Alignment-series correlation with output | Maximum displacement RMS at step |
|---|---:|---:|---:|
| First convolution | 11 / 61 | .7637 | 415 |
| Residual stage 1 | 10 / 61 | .8279 | 416 |
| Residual stage 2 | 14 / 61 | .9052 | 416 |
| Residual stage 3 | 18 / 61 | .9945 | 416 |
| Pooling | 20 / 61 | .9945 | 416 |
| Log-probability output | 21 / 61 | 1 by definition | 416 |

- **Central message:** the large excursion has a distributed functional expression: representation motion reverses and increases in amplitude, with deeper-stage alignment more closely tracking the output.
- **Timing markers:** first negative output-change cosine within the display is **391**, following a nonnegative value at 390. The selected event first rises at **405**, crosses +10% at **406**, and peaks at **415**. The early negative observation is not a unique onset or a validated warning; separate small reversals occur before the major excursion.
- **Amplitude:** first-convolution movement reaches its window maximum at the loss peak; the other five endpoints reach theirs one update later. This is an observed association, without a claim about causal propagation or a universal layer of origin.
- **Coverage:** one seed, event and probe; the probe is 1,000/50,000 training images. Full and probe loss remain separate curves. Correlations are descriptive and temporally dependent.

## Verification and scope

- Raw boundary file: 5,068,541 bytes, SHA-256 `f26850a016d3fb980033e3736ee722449f74e6eded760d968f1888b9179621e2`, matching the retained transfer inventory.
- All **313 shared loss states** exactly reproduce the native Run223 control; **48** overlapping states exactly match the prior Run243 trace. Across the six common stages, **372** cosine/RMS values agree within **6.67e-16**. Source manifests match data, initialization and parent-control identities.
- This continuation is **not bitwise replay of Run215's original overview**. Events are recomputed on the actual continued loss, using the unchanged detector; old event coordinates are not transplanted from that other trajectory.
- Original same-hardware parameter-triplet replay independently checked representation reductions with maximum absolute error **1.78e-15**. The new work checks retained source integrity, indexing, selection and plotted values; it does not claim a new dense activation capture.
- **No training, cloud execution or new run ID:** the required every-step measurements already exist. The previous 280--310 outgoing-window SVD, all-stage report and their conclusions remain preserved. Their ResNet spectra/overlap do not describe 390--450.

## Figure, caption and reproduction

- [Updated three-model figure](../../../manuscript/consolidation/evidence/05-adam/b-tracing/figures/00-three-model-loss-representation-motion.pdf).
- [Figure generator](../../../manuscript/consolidation/work/resnet_peak_revision_20261114/01_figure.py); [verification](../../../manuscript/consolidation/work/resnet_peak_revision_20261114/02_verify.py); [current captions](../../../manuscript/consolidation/evidence/FIGURE_NOTES.md).
- Caption: Loss and functional motion across three unchanged-Adam continuations. Columns show MLP, ResNet and Transformer; rows show full/probe loss, consecutive representation-change cosine and displacement RMS divided by each stage's own window maximum. Each change is placed at its observed completed step. Windows are 451--501, 390--450 and 553--601. The ResNet window contains the +311.53% excursion peaking at 415 and recovering at 439. Blue lines mark the first observed negative output-change cosine in each display; MLP is already negative at entry. Black lines mark the first full loss at least 10% above the pre-peak minimum. These selected windows describe participation and timing; the separate SVD companion retains its original ResNet window 281--311.
- Reproduce the reduction, figure, caption/catalog update, figure verification and evidence package in that order. All previous figures and intervention assets are retained unchanged; publication output is PDF and scratch renders remain ignored.
