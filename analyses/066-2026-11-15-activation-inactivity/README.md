# Activation distributions and inactive units

**Status: complete and verified.** Start with [executive findings](FINDINGS.md), [visual summary](activation-summary.pdf), or [all-site activation matrices](activation-matrices.pdf). No manuscript edits or Runpod usage.

- MLP: no wholly inactive tanh units; saturation and energy concentration are different measurements.
- ResNet: 50%/75% inactive post-add channels in groups 2/3 predate treatment and follow the zero-initialized residual/zero-padded shortcut construction; 90 saved block-state audits confirm the matching parameter/gradient zeros.
- Transformer: approximately 85% zero entries but no wholly inactive FF units across all training positions. Caps substantially reduce final-layer energy dominance without reviving dead units.
- Completed: 21 full-training passes, 48 reference-state measurements, 1,104 site/scope observations, 23,184 threshold rows, and 21 rendered PDF pages. Capture took about 5.5 minutes on owned Spark, excluding preparation/transfer/analysis; zero cloud cost.

## Question and design

- Compare unchanged Adam, distributed cap q25, and cap q50 using actual retained states from Runs 293-295. One matched initialization per model.
- Six stored states: fork, fork+1, fork+32, fork+33, end-1, end. Windows are 512-800 (MLP), 256-576 (ResNet), 544-928 (Transformer). The shared fork is measured once and checked identical across arms.
- All training examples at fork, fork+32, and end: 5,000 MLP images; 50,000 ResNet images in original 1,000-image normalization groups; 59,680 Transformer sequences / 2,088,600 valid target positions. No padding coordinates count.
- Original reference sets at all six states: all MLP images, first intact ResNet group, first 64 Transformer sequences / 2,240 token positions. These provide sparsely sampled temporal comparisons, not every-update histories or peak localization.
- No optimizer updates: replay saved models through forward computation. Qualify actual forward/backward training and no-gradient evaluation on the same reference batch, including exact hooked/plain gradients. Train/eval equality is checked rather than presumed; ResNet has no running BN statistics, and Transformer dropout is zero.

## Operational definitions

- Matrix rows are sample/position pairs; columns are features/channels. A CNN channel is one unit; it must be inactive at **all spatial locations in all covered images** to count as inactive.
- **Inactive unit at threshold epsilon:** `max(abs(A[:, j])) <= epsilon`. Report exact-zero separately. This is observed inactivity, not proof that a unit cannot reactivate on other inputs or future steps.
- **Weak unit:** unit RMS below epsilon. Unlike the maximum rule, this permits rare responses; retain both.
- **Sparse entries:** fraction of individual matrix entries below threshold. Many zero entries do not imply many wholly inactive units.
- Sweep absolute thresholds and thresholds relative to the common fork's full-training layer RMS, fixed across states and arms. Do not independently normalize each treated state.
- **Concentration:** per-unit squared-activation shares, top 10% of units' share, and effective unit fraction `1 / (sum(shares**2) * width)`. This is coordinate energy concentration, not a matrix rank or causal importance measure.
- Tanh saturation `abs(A)>=0.99` is separate from near-zero inactivity. Signed activations use absolute magnitude; negative values are not called inactive.
- Sites: all MLP affine/tanh layers; every distinct ResNet ReLU invocation, pooling and readout; Transformer embedding, position-added representation, both attention residuals, FF preactivations, FF ReLU outputs and block outputs. Vocabulary readouts are excluded from neuron-inactivity metrics; near-zero word probabilities would not mean dead hidden neurons.
- Activation matrix displays retain all unit columns and at most 4,096 deterministically spaced rows from the first physical batch. They are illustrative subsets; statistics use all declared rows, without sample/spatial averaging. Original row and unit order is retained or explicitly labeled if sorted.

## Sources, execution, and deliverables

- Scientific worker/config: [Run 296](../../runs/296-2026-11-15-checkpoint-activation-audit/README.md). New analysis and diagnostic execution explicitly authorized by the user on 15 November 2026.
- Raw compressed per-unit statistics, exact threshold counts, log-magnitude histograms, and illustrative matrices are retained under `raw/` locally and on Spark. Model/data/checkpoint source hashes are recorded. No dense activation history is retained.
- Deliverables: executive findings, layer/threshold tables, temporal inactivity and concentration heatmaps, final-state distributions, and a complete per-site activation-matrix atlas, all in this folder.
- Coverage is one seed and selected saved states. No significance tests treating neurons or steps as independent model replications; no generalization, pruning benefit, or cause-of-spikes claim.
