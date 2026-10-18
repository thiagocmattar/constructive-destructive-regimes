# Analysis024: one-initialization, 3k GD learning-rate calibration

The user requested two additional learning-rate conditions per case on owned
DGX Spark, with 3,000 full-GD updates and one frozen initialization. This
authorizes only the five new runs below. Analysis020/021 queues stay held;
no cloud rental or manuscript work. Executed inputs remain unchanged.

**Status October18 at06:44 Sao Paulo time:** Runs122/123 (MLP0.04 and
ResNet0.025) are complete, retrieved and verified. Run124 (ResNet0.05) is at
2,456/3,000 updates; Transformer0.1/0.2 remain queued. Six of nine total
conditions are verified, including four reused sources. Remaining ETC is54.90h
(65.88h with20% allowance), projecting October20 around13:40 Sao Paulo time.
The completed0.025 ResNet first reverses at gradient update1,121, later than
231 at0.1; its final train loss is1.466540 and B/A0.138530.
Launched October17 at09:13. See `launch.json`, `status_update_2026-10-18.json`
and [available results](observations/001-available-calibrations.md).

| Dataset / model | LR | Evidence | Updates |
| --- | ---: | --- | ---: |
| CIFAR-10 first5k / tanh200-200 MLP | 0.004 | Reuse Run059 prefix | 3,000 |
| Same MLP | 0.02 | Reuse Run062 prefix | 3,000 |
| Same MLP | 0.04 | New Run122 | 3,000 |
| CIFAR-10 all50k / ResNet-32 | 0.025 | New Run123 | 3,000 |
| Same ResNet-32 | 0.05 | New Run124 | 3,000 |
| Same ResNet-32 | 0.1 | Reuse Run120 | 3,000 |
| WikiText-2 / two-layer Transformer | 0.05 | Reuse Run121 | 3,000 |
| Same Transformer | 0.1 | New Run125 | 3,000 |
| Same Transformer | 0.2 | New Run126 | 3,000 |

## Selection and interpretation

MLP LR0.004 has no negative adjacent gradient within3k. Existing LR0.01 first
reverses at update4,565, so it would not provide a transition in this window.
LR0.02 first reverses at1,156; reuse those exact3k values and test0.04 for an
earlier transition. ResNet LR0.1 first reverses at231. The proposed0.025/0.05
factor-of-two reductions test whether reversal is delayed while remaining
inside3k; that timing is a hypothesis, not an inverse-LR prediction. Transformer
LR0.05 has no reversal within3k, while an earlier500-update LR0.1 control first
reversed at456; test0.1/0.2 through3k. No transition is guaranteed at a new rate.
Rates are selected using prior training evidence, not test metrics. Failed or
nonfinite conditions remain visible; no replacement rate is launched automatically.
The legacy MLP transition indices are recorded in the
[earlier calibration estimate](../022-2026-10-15-eos-long-horizon-plan/estimate.json).

The completed0.04 pilot has an initial negative-cosine transient (first pair at
gradient update2), followed by positive alignment and a later negative phase.
Its first100-pair window with at least80% negative pairs ends at323; the first
all-negative100-pair window ends at343. These descriptive criteria distinguish
the transient from persistent reversal, without establishing a stability regime.
See `05_check_mlp_transitions.py` and `mlp_transition_check.json`. Keep this
calibration caveat when reviewing the proposed rates for the final figure.

GD removes mini-batch sampling noise here, but initialization still matters.
Use the same frozen seed0 tensors across rates within each case; the user has
chosen to prioritize rates over initialization replication. Results therefore
support an LR comparison conditional on this initialization, not robustness
across initializations. These are three task/model cases and two datasets.

## Matched setup and diagnostics

Reuse unchanged qualified Analysis020 scientific code and frozen inputs, as
in [Analysis023](../023-2026-10-15-eos-smallest-lr-calibration/README.md):
plain GD, constant LR, no momentum, decay, clipping, schedule, augmentation or
dropout; FP32 parameters/gradients, TF32 disabled and FP64 diagnostic geometry.
One update processes5k MLP images;50k ResNet images averaged over the same50
fixed1k ghost-BN groups; or59,680 Transformer sequences /2,088,600 target tokens.
Dataset/token weighting, exclusions, physical accumulation and loss are fixed.

Retain unsmoothed full-objective loss, every raw-gradient norm and adjacent
cosine, q=eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous), A=sum(norm(g)),
B=norm(sum(g)), B/A, energy, all-pair/lag-one/beyond-lag-one interactions and
actual-update/displacement diagnostics. First/zero-denominator pairs remain
undefined. q is a projected finite-step gradient change, with a Taylor
remainder in a local-curvature interpretation; it is not lambda_max.

Largest algebraic Hessian eigenvalue at states0,100,...,3000, independently
checked relative residual<=0.005. MLP uses all5k images; ResNet uses its first
five fixed ghost groups; Transformer uses its first2,500 sequences. Larger-model
spectra remain subset estimates, not full-objective sharpness. Legacy MLP
spectra requested two eigenvalues; new runs request one. Compare the largest
in both, disclose this solver difference, and retain all residual checks.

Keep recovery every100 and retained states500/1000/2000/3000 plus final,
including optimizer/RNG/data/diagnostic continuation state. Full test evaluation
every500 and at the final state; no LR selection from test results. Metrics
cover gradient states0..2999; final objective and spectrum cover state3000.

## Runtime, storage and execution bounds

Measured complete3k runs on this Spark took43,220.45s for ResNet (12h00m20s)
and94,955.23s for Transformer (26h22m35s), including training-time diagnostics
and checkpoint/evaluation work. Four new large-model runs project76h46m;
allow3min for the new MLP/setup, giving about76h49m (3d4h49m). A20% allowance
gives92h11m (3d20h11m). These are qualified estimates, not confidence bounds:
new learning rates can change eigensolver convergence and throughput.

Hard queue limit120h. Per-run bounds are1h for MLP and1.5x the measured
large-model duration plus30min. Do not restart executed conditions or admit
unrelated jobs. Five new runs execute sequentially:122,123,124,125,126; then
the queue exits. No predecessor continuation or later training is authorized.
New runs keep the unchanged3k recipe; the previous500-step Transformer0.1
run is calibration evidence, not an assumed resumable3k trajectory.

Reserve15GB additional retained evidence on each machine. The read-only
October17 preflight found about3.71TB remote /216GB local free and no GPU
compute process. Rental cost is zero on user-owned compute. Retain raw/source
evidence on both machines; remove only verified redundant transfer archives.
Verify sizes/SHA256 inventories before reducing or plotting.

Detached remote execution survives SSH disconnection and persists logs,
manifests, recovery states and progress events. The hidden local monitor reads
every30min, records steps/loss/throughput/memory/GPU temperature/power and
refreshed ETC, retrieves terminal new runs, and verifies available results.
Monitoring/retrieval do not admit training. A failed condition is reported
without silently replacing its rate or hiding the failure.

## Reviewed MLP preview

The [MLP three-rate figure](figures/eos_mlp_learning_rates_3k.pdf) overlays
0.004/0.02/0.04 through3,000 updates: unsmoothed loss, eta*lambda_max,
unsmoothed eta*kappa_hat and raw-gradient norms (log Y). All27,093 plotted
points,8,997 pairs,93 full5k spectra and62 source-inventory files pass checks;
the current four-panel Poppler render is reviewed. See
[observation002](observations/002-mlp-learning-rates.md), `06_plot_mlp.py`,
`mlp_learning_rates_results.json` and `mlp_learning_rates_figure_manifest.json`.
This local preview preserves the frozen packet, training queue and existing PDFs.

## Final figure plan and records

After calibration review, the final overview retains three model columns and
four rows: unsmoothed training loss, eta*lambda_max, unsmoothed eta*kappa_hat,
and raw-gradient norms with log Y. Overlay three LR colors per column, with
explicit legends and subset-spectrum labels. Keep the q=1 reversal and
eta*lambda=2 quadratic reference lines; neither establishes global stability.
Raw adjacent cosine and normalized displacement remain available for companion
figures. No manuscript insertion is performed by this calibration pipeline.

Scripts: `00_prepare.py` freezes configs/source inventory and verifies reused
evidence; `01_queue.py` uses the qualified Analysis023 executor with a strict
five-ID admission list; `02_launch.py` checks the remote packet and idle GPU,
then detaches; `03_results.py` retrieves/verifies terminal runs and reduces
all available3k conditions; `04_monitor.py` performs bounded hidden reads and
automatic retrieval. `cohort.json`, `preflight.json`, `remote_preflight.json`,
`launch.json` and owning-run receipts record execution. `results.json` and
[observation001](observations/001-available-calibrations.md) distinguish
available from planned conditions. No final figure is claimed before review.
The separate post-launch `05_check_mlp_transitions.py` only reduces retained
MLP evidence; it does not change the frozen training packet or admit new work.

Next Run127 / Analysis025 / H021. H020 remains the scientific question.
