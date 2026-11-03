# Analysis024: one-initialization, 3k GD learning-rate calibration

The user requested two additional learning-rate conditions per case on owned
DGX Spark, with 3,000 full-GD updates and one frozen initialization. This
authorizes only the five new runs below. Analysis020/021 queues stay held;
no cloud rental or manuscript work. Executed inputs remain unchanged.

**Completed October20 at13:47 Sao Paulo time:** all nine conditions reached
3,000 full-GD updates. All outputs were retrieved and verified at14:47:
254 source-inventory files,26,991 adjacent pairs and279 spectral states pass.
The final Transformer LR0.2 run has training loss6.069747, test loss5.858303,
first negative gradient update10, last1k mean cosine-0.9936598 and final
B/A0.05413294. Training/retrieval ETC is zero; the owned Spark is idle.
The five-run queue took76.56h and stopped without predecessor continuation.
A monitor SSH error delayed final retrieval; the direct connection succeeded,
the task monitor was stopped and manual `03_results.py --collect` passed.
The five new runs retain6.395GB in160 inventoried files on each machine;
verified redundant transfer archives were removed. No cloud rental.
See [completion observation](observations/004-completed-calibration.md) and
`completion_verification.json` for the closeout evidence.
The three ResNet rates0.025/0.05/0.1 first reverse at gradient updates
1,121/494/231, with final train losses1.466540/0.967229/0.489838.
Launched October17 at09:13. See `launch.json`,
`status_update_2026-10-18_1305BRT.json` and
[available results](observations/001-available-calibrations.md).

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

## Reviewed completed-run overview

The user expanded the [existing PDF](figures/eos_mlp_learning_rates_3k.pdf)
to three model columns and four rows. The October20 update includes all nine
completed conditions: MLP0.004/0.02/0.04, ResNet0.025/0.05/0.1 and
Transformer0.05/0.1/0.2.
Rows show unsmoothed loss, eta*lambda_max, unsmoothed eta*kappa_hat and
raw-gradient norms (log Y). Transformer0.2 is green, alongside the existing
blue0.05 and orange0.1 curves, through all3,000 updates.
Full/subset Hessian labels distinguish the measurement coverage.

All81,279 plotted points,26,991 pairs,279 spectra and254 source-inventory
files pass checks; the current twelve-panel Poppler render is reviewed. All32
previously plotted curves are unchanged. See
[observation003](observations/003-completed-three-model-overview.md),
`06_plot_mlp.py`, `mlp_learning_rates_results.json` and
`mlp_learning_rates_figure_manifest.json`. Filenames retain the original MLP
prefix at the user's request; metadata explicitly states the expanded scope.
This local figure update preserves the frozen packet, queue and other PDFs.

The user subsequently authorized a manuscript experimental section. The
[IEEE revision and integration](observations/006-ieee-figures-and-manuscript.md)
updates the six isolated views, adds two compact main-text composites, and
inserts Section IV and a raw-alignment Appendix D into the working paper.
See the [critical figure review](FIGURE_REVIEW.md). Artwork is 7.16 inches wide,
has no embedded footers, and places panel letters outside the axes. All 54
standalone Y arrays are unchanged; gradient X indices now follow manuscript t.
`08_build_manuscript_assets.py` copies reviewed figures and generates all nine
endpoint rows, with hashes in `manuscript_assets.json`. All scientific training
inputs and the original overview remain preserved; this extension is local
postprocessing and manuscript authoring, with no new run or cloud execution.

## Final figure and records

The reviewed final overview retains three model columns and four rows:
unsmoothed training loss, eta*lambda_max, unsmoothed eta*kappa_hat, and
raw-gradient norms with log Y. Three LR colors per column have explicit
legends and subset-spectrum labels. Reference lines mark q=1 reversal and
eta*lambda=2 for a quadratic; neither establishes global stability.
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
available from planned conditions; all nine are now complete. The final
figure passes source/array/hash checks and visual review.

## Isolated three-column figures, October30

The requested companion views each contain one row and three model columns,
with all nine completed conditions, consistent LR colors and solid data curves:

- [Training loss, log Y](figures/eos_gd_loss_log_3k.pdf), unsmoothed.
- [Consecutive cosine](figures/eos_gd_rho_3k.pdf), unsmoothed.
- [Consecutive cosine, smoothed](figures/eos_gd_rho_smoothed_3k.pdf).
- [Gradient-derived eta*kappa_hat, smoothed](figures/eos_gd_eta_kappa_hat_smoothed_3k.pdf).
- [Gradient norms, smoothed, log Y](figures/eos_gd_gradient_norms_smoothed_3k.pdf).
- [Normalized displacement R_t](figures/eos_gd_normalized_displacement_3k.pdf), unsmoothed.

Smoothing is a 50-observation trailing arithmetic mean, using complete windows
only and no edge padding. Norms are averaged before log display. There is no
seed aggregation. The generator `07_plot_single_rows.py` checks used-source
hashes against the independently verified results, config/initial identities,
q and B/A identities, all smoothed values against independent convolution, and
every plotted artist array. All six PDFs pass page/text/bounds and Poppler
visual review. The 54 curves contain 160,659 plotted points. Ignored renders
and arrays remain under `scratch/` and `derived/`; compact provenance is in
`single_rows_results.json` and `single_rows_figure_manifest.json`.
See [observation005](observations/005-isolated-three-column-figures.md) for
indexing, captions, coverage and interpretation limits. Existing figures and
executed evidence are preserved; this is local postprocessing only.
The separate post-launch `05_check_mlp_transitions.py` only reduces retained
MLP evidence; it does not change the frozen training packet or admit new work.

## Second critical review of manuscript Section IV

[Observation007](observations/007-section-iv-critical-review.md) reviews the
scientific argument and adds a reproducible audit of retained scalar evidence.
All nine final all-pair interactions are positive, while seven cumulative
lag-one interactions are negative. Absolute displacement rises with LR as R
falls within each case. A correctly indexed retrospective check quantifies
the local loss approximation's limitations. `10_review_section_iv.py` and
`section_iv_review_checks.json` preserve definitions, coverage and hashes.
The draft, figures, executed inputs and raw evidence are unchanged.

## Applied manuscript revision

[Observation008](observations/008-applied-section-iv-revision.md) records the
user-authorized application of all eight surgical edits. Section IV now
reports the local/all-lag distinction, absolute displacement and retrospective
loss audit; Appendix D contains detailed methods and the three supporting
tables. All 36 table rows and 26,991 loss comparisons pass independent checks.
The 18-page draft is compiled and visually reviewed. Existing scientific
figures and executed evidence are preserved; this remains local postprocessing.

Next Run154 / Analysis026 / H022 (see research/INDEX.md). H020 remains the scientific question.

## Manuscript-only revision, 2026-11-02

The [author-feedback observation](observations/009-author-feedback-figures.md) records the new compact publication
view and its unchanged source evidence. Reproduce the current manuscript
through `manuscript/draft/revisions/2026-11-02-feedback/`; earlier paper
validators refer to their historical snapshots. No training or cloud work.

## Author-feedback spectral comparison, 2026-11-03

[Observation010](observations/010-curvature-spectral-overlay.md) records the
new Figure 1 overlay of 279 retained eigenvalue measurements on the unchanged
loss and gradient diagnostic curves. Coverage and indexing are explicit;
all 53,838 plotted values are checked against retained raw records.
The new PDF preserves the previous figure under its existing filename.
The [current manuscript revision](../../manuscript/draft/revisions/2026-11-03-original-argument-feedback/README.md)
also removes the endpoint table from compilation while retaining its source.
No new training or Hessian computation.
