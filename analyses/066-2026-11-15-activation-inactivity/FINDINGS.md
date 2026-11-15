# Activation audit: what we learned

**Completed. Start with [the seven-page visual summary](activation-summary.pdf).**
The [matrix atlas](activation-matrices.pdf) shows every measured site, with all three arms side by side. The manuscript was not changed.

## Main findings

- **Sparse activations are not the same as dead units.** At the final Transformer state, 84.7% of entries in its second FF ReLU are exactly zero under Adam, but **none of its 200 units is inactive across all 2,088,600 training positions**. q25/q50 give 83.1%/83.2% zero entries, also with no completely inactive units.
- **Transformer energy concentration changes substantially under the caps.** At that same final site, the strongest 10% of units carry 49.1% of activation energy under Adam, versus 33.1%/33.8% under q25/q50. The effective number of units is 11.3/84.4/76.3 out of 200. This is inverse concentration of squared activation, **not SVD rank**.
- **A single Transformer unit explains much of this difference.** At t=928, unit 37 carries 28.90% of the site's full-training activation energy under Adam, versus 5.49%/6.79% under q25/q50. At the common fork, the strongest unit carries only 4.66%. This endpoint comparison identifies a concrete coordinate to inspect next; it does not prove that this unit causes spikes or that the cap works by preventing its dominance.
- **MLP: no dead or near-zero hidden units in the measured full-data states.** Both tanh layers have zero wholly inactive units, including at a threshold of 1% of the common-fork layer RMS. The first tanh layer has nearly uniform energy across units (effective fraction about 99%); its large fraction of saturated entries is a separate phenomenon: 51.5% under final Adam versus 49.1%/49.2% under q25/q50. The second layer's strongest 10% carry about 22% in every arm.
- **ResNet: a fixed inactive-channel pattern predates the interventions.** Every post-addition ReLU in residual group 2 has channels 16-31 exactly zero; group 3 has channels 16-63 zero. These are 50% and 75% of the respective residual-stream widths. They are inactive over all 50,000 training images and spatial positions, at all three full-data checkpoints, under all three arms. The within-branch ReLUs have no completely inactive channels in these checks.
- **The ResNet pattern has a specific structural explanation.** The implementation zero-initializes each residual branch's final BatchNorm scale and pads widening shortcuts with zeros. Added channels then enter the post-addition ReLU at zero, whose saved full-gradient readouts are also zero on these BN coordinates. A separate 90-block-state audit confirms zero scale, bias, and their gradients on exactly the inactive channels. This is not an inactivity increase caused by Adam spikes or an effect removed by the caps. It is a relevant architecture/initialization caveat for interpreting this ResNet experiment.

## Compact final-state comparison

Complete training set; last hidden nonlinearity in each model. Percentages are rounded; the CSVs preserve full precision.

| Model / site | Arm | Zero entries | Completely zero units | Strongest 10% energy | Effective units / width |
|---|---|---:|---:|---:|---:|
| MLP / Tanh 2 | Adam | 0.0% | 0.0% | 22.1% | 155.5 / 200 |
| MLP / Tanh 2 | q25 | 0.0% | 0.0% | 22.3% | 154.5 / 200 |
| MLP / Tanh 2 | q50 | 0.0% | 0.0% | 22.3% | 154.9 / 200 |
| ResNet / final post-add ReLU | Adam | 78.7% | 75.0% | 52.0% | 15.3 / 64 |
| ResNet / final post-add ReLU | q25 | 78.5% | 75.0% | 52.2% | 15.3 / 64 |
| ResNet / final post-add ReLU | q50 | 78.5% | 75.0% | 52.2% | 15.3 / 64 |
| Transformer / Block 2 FF ReLU | Adam | 84.7% | 0.0% | 49.1% | 11.3 / 200 |
| Transformer / Block 2 FF ReLU | q25 | 83.1% | 0.0% | 33.1% | 84.4 / 200 |
| Transformer / Block 2 FF ReLU | q50 | 83.2% | 0.0% | 33.8% | 76.3 / 200 |

## What this says about the paper's interpretation

- Low-dimensional **representation changes** do not imply that only a few **neurons activate**. They are different measurements: temporal movement patterns versus coordinate-wise activation occupancy/energy.
- These data do not support a general story in which the cap helps by reviving dead neurons. Exact inactivity is unchanged here; Transformer energy distribution is the more promising distinction.
- The ResNet finding should be considered before making a broad statement about utilization of all residual-stream channels. The controlled cap-versus-Adam comparison still uses the same architecture and initialization in every arm.
- This is descriptive evidence at selected saved states. It does not establish behavior between checkpoints, permanent death, generalization, pruning benefit, or a unique cause of spikes. No manuscript claim was changed automatically.

## Coverage and definitions

- **Arms:** unchanged Adam, cap q25, cap q50; actual saved tensors from Runs 293-295, same initialization and fork per model.
- **States:** MLP 512/513/544/545/799/800; ResNet 256/257/288/289/575/576; Transformer 544/545/576/577/927/928. Shared forks are measured once, verified identical, and reused explicitly.
- **Full data:** fork, fork+32, end; 5,000 MLP images, 50,000 ResNet images in intact original BN groups, 59,680 Transformer sequences with 2,088,600 valid token positions. This gives 21 full-data passes across unique model states.
- **Reference data:** six states per arm with shared forks; 48 unique measurements using 5,000 / 1,000 images / 64 sequences. All activation samples/positions enter the statistics; no spatial or sample averaging precedes inactivity tests.
- **Sites:** 5 MLP, 33 ResNet (including separately identified repeated ReLU calls), 10 Transformer. Interpret dead-neuron counts primarily at the nonlinearities; affine/residual/readout measurements are also retained.
- **Inactive:** maximum absolute activation over all covered rows at most epsilon. **Weak:** RMS at most epsilon. **Sparse entries:** fraction of coordinates below epsilon. Absolute and fixed-fork-relative threshold sweeps are retained. Extended relative thresholds 3/10/30/100 complete the plotted distributions; they are not claims about small activations.
- **Matrices:** all feature/channel columns retained, at most 4,096 deterministic position rows stored from the first physical batch and 512 displayed. These are illustrative subsets; the numerical summaries use full declared coverage. Per-site color limits match across arms and clip the pooled 99.5th magnitude percentile only for display.
- **Mode checks:** at each model's common checkpoint, hooked/plain backward gradients agree exactly and activations match exactly between gradient-enabled, no-gradient, and eval-mode forwards. Models use zero dropout and ResNet has no running BN buffers. Statistics come from train-mode forward passes at the stored training states; no new optimizer steps were needed.

## Verification and reproducibility

- Run 296 synthetic tests check absolute-value handling, zero versus weak units, threshold equality, streaming moments, energy concentration and zero denominators.
- Independent reduction checks 1,104 site/scope observations, 23,184 threshold rows, counts/coverage, shared forks, and all 90 structural audit rows.
- 24 ResNet/Transformer checkpoint hashes match the original retention inventories. All downloaded output files match remote byte counts and SHA-256 hashes. Dataset/reference source hashes match the original run manifests.
- Recomputed full/reference losses agree with historical losses to maximum relative error 3.78e-6 across GPU/runtime differences; no claim of historical bitwise trajectory replay is made.
- Capture used owned Spark only: approximately 3.6 s MLP, 189.7 s ResNet, and 139.5 s Transformer (about 5.5 min total, excluding preparation, qualification retries, transfer, plotting and review). Peak allocated CUDA memory: 0.10/4.39/1.81 GB. No Runpod usage, zero cloud cost, no training-state modification.
- One early mechanics attempt lacked staged run manifests; another detected nondeterministic convolution-gradient replay. Both are excluded. Deterministic cuDNN enabled before the successful qualification and all production measurement; failed smoke evidence remains in ignored remote scratch directories.
- Reproduce: `01_collect.py` retrieves frozen output; `02_reduce.py` verifies and exports tables; `03_plot.py` regenerates both PDFs; `04_resnet_structure.py` performs the separate saved-parameter audit; `05_render.py` renders and checks publication artifacts; `06_verify.py` verifies the report claims, frozen inputs, and compiled numeric table.

## Files to use

- [Visual summary](activation-summary.pdf): 7 pages; final comparisons, threshold sweeps, depth, distributions, and sparse temporal maps.
- [All-site matrix atlas](activation-matrices.pdf): 14 pages; MLP pages 1-2, ResNet 3-11, Transformer 12-14.
- [Layer summaries](layer-summary.csv), [threshold sweep](threshold-sweep.csv), [persistent/ever inactivity across sampled states](temporal-inactivity.csv), [coverage](coverage.csv), [structural audit](resnet-structure.json).
- Raw per-unit arrays and matrices (about 587 MB) are in ignored `raw/`, also retained on Spark at the path in `launch.json`. Worker processes have exited. No dataset/checkpoint copies were added to Git.

## Next questions, not executed

- Trace Transformer unit 37 around the loss excursions using denser saved/replayed states, including activation tails and its downstream contribution. The present endpoint contrast does not resolve when dominance emerged.
- If the ResNet construction is changed, treat nonzero residual-branch initialization or a different widening shortcut as a new matched experiment. Do not silently substitute it into the existing paper evidence.
