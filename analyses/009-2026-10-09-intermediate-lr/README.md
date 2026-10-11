# Analysis009: four intermediate warmup-only learning rates

Status: all four full passes completed and passed scientific verification at
2026-10-09 18:36-18:37 UTC. Each covered all712 updates and1.492B input tokens,
with zero overflow retries. The36 completed compact metric files passed local
size/SHA256 verification. [Seven-rate results and seven reviewed PDFs](complete-seven-lr/README.md)
are ready. The best new condition is LR0.004641588834, validation loss4.318091;
LR0.002154434690 follows at4.410370. Recovery downloads and cloud cleanup remain
in progress. The 18:45 UTC read found all GPUs idle at31-34 C.
The user explicitly approved compute plus temporary checkpoint storage within
USD15. Two task-owned workers and one 60GB network volume are active in EU-RO-1.
The user requested two points between each pair of the three completed rates.
Resolved as geometric spacing: 10^(-11/3), 10^(-10/3), 10^(-8/3), 10^(-7/3),
approximately0.000215443469,0.000464158883,0.00215443469,0.00464158883.

Question: how do loss and raw-gradient, stored Adam momentum, and adaptive
direction geometry behave between the original LR conditions? This is an
exploratory single-seed comparison, with no prescribed monotonicity or cosine
threshold for convergence. Negative cosine means negative interaction between
the particular vectors, without establishing stability or a loss mechanism.

## Matched comparison and coverage

Reuse the exact Run021 training implementation and frozen Run002 initial tensors;
seed1234 token order and identical pinned MiniPile cache. FP16 compute, FP32
parameters/moments, Flash SDPA, batch1024,microbatch4,length2048,AdamW(0.9,0.95),
epsilon1e-8,weight_decay0.1,clip1.0. Warmup is the existing pre-step1% ramp;
LR0 at update1 and constant peak from update9. LR also scales weight decay.
712 updates,728375 sequences,1491711416 inputs and1490983041 supervised targets,
with no repeats or discarded tail. Initial/final validation includes all339
sequences/693329 targets and uses token weighting.

For g (raw preclip), m (actual uncorrected exp_avg after moment update), and d
(bias-corrected preconditioned direction excluding LR and weight decay), retain
norm,A=sum norms,B=norm sum,B/A,adjacent cosine at every accepted update plus
paired norm ratios. Use the [Analysis008 contract](../008-2026-10-08-no-decay-momentum/README.md)
unchanged. Undefined denominators are null with reasons. Recoveries initially,
every100 updates, at356 and finally retain model, optimizer, scaler, RNG/order
position and all geometry buffers. Longer-lag measurements remain exact sparse
checkpoint comparisons; no dense temporal history is implied.

Four independent processes run concurrently on four GPUs, one per GPU. Pin library
versions to Torch2.11.0/Transformers5.12.1/NumPy2.5.0 and CUDA13 wheels; record
actual runtime. The old three conditions ran on ARM/GB10 Spark; architecture,
GPU kernels and floating-point trajectories differ on Runpod even with equal
initial tensors. This limits a seven-point pooled causal comparison. Do not
rewrite original inputs or imply independent-seed uncertainty.

## Execution and artifact plan

Operational ceiling USD15 aggregate. Two workers each have two RTX4090 GPUs
at USD1.78/hour per worker; fleet rate USD3.56/hour. The absolute deadline is
2026-10-09 20:30 UTC (17:30 Brasilia), including setup, all calibration attempts,
training, validation and retrieval. From actual worker creation times, the GPU
cost bound is USD12.5411; a USD0.15 storage allowance gives USD12.6911 maximum
estimated aggregate cost. Provider billing remains provisional. The 60GB network
volume costs USD4.20/month if retained. Container disks are 50GB per worker.

| Run | Peak LR | Worker | Physical GPU | Full wrapper PID |
| --- | --- | --- | --- | --- |
| 024 | 0.000215443469003 | mtd2dlb7w1mslf | 0 | 1798 |
| 025 | 0.000464158883361 | mtd2dlb7w1mslf | 1 | 1847 |
| 026 | 0.002154434690032 | kykdwoonqs0z9s | 0 | 615 |
| 027 | 0.004641588833613 | kykdwoonqs0z9s | 1 | 664 |

Worker creation times are 16:50:52 UTC and 17:06:23 UTC. Volume mja1prpve4 is
dedicated to this cohort. Disjoint run directories prevent concurrent output
collisions; frozen shared inputs are read only, with token data staged locally.
The mutable lease/placement supersedes prepared execution metadata in the frozen
configs, including the former two-processes-per-GPU/three-hour estimate;
scientific configs/code remain unchanged and hash frozen. Source packet SHA256
is b9701ba9dc34952e422f1fc248a5e6c8a2c0f2b31322026eb0f46dbaed2ed14b.
Network volume60GB holds shared data, approximately29GB of full-pass artifacts
and the preserved calibration states. Live
pricing/catalog reads and created resource attribution are recorded separately.
The official PyTorch template is chosen from the live catalog. No foreign
resources are modified. Calibrate12 updates per run, including four plateau
updates, then reset to frozen initialization for the full pass. Qualify runtime,
source/input identity, all metrics and lease fit before full launch.

Run detached, persistent logs, manifest/events and wrapper states. Monitor every
60seconds; refresh throughput and ETC, check event age, temperature, memory,
disk, finite loss and overflow retries. A falling loss or positive alignment
alone is not a training-health verdict. A local cost guard stops compute at
the deadline; network state survives. Automatic monitoring needs this computer
awake and connected. Retrieve each run, verify every file size andSHA256 before
removing its worker, then remove the task-owned network volume after all four
retrieval gates pass. If retrieval fails, preserve durable evidence and report
ongoing storage billing. Never delete scientific evidence to meet a deadline.

Scripts:00_prepare.py freezes configs/source;01_worker.py performs bounded
calibration/full execution;02_verify.py checks scientific completeness and
state;03_setup.sh pins runtime and stages verified data locally;04_operate.py
performs native SSH/SCP and measured qualification;05_watch_retrieve.py monitors
and retrieves;06_deadline_guard.py independently stops owned compute on expiry.
07_fetch_result_metrics.py verifies completed compact inputs independently of
large recovery transfers;08_retrieve_bundles.py packs remaining immutable files
on container disks, skipping accepted files;09_analyze_results.py produces the
seven-rate metrics and cumulative geometry PDFs. Full transfer receipts remain
mandatory before worker/volume deletion. The current hidden controller PID is49480;
independent deadline guard PID40476. Both use the original20:30 UTC deadline.
The original Analysis008 PDFs are preserved. New geometry has two pages (four
then three columns), with an independent linear top-row y-scale per LR.
Coarse checkpoint lag profiles remain pending local recovery availability.

The requested figure revision uses10_plot_lr_geometry.py: panel(e) raw norms
on logarithmic Y, plus a new2x2 cumulative LR view. Both use reversed plasma,
with color lightness decreasing as LR increases; numerical results are unchanged.

11_plot_loss_ratio_alignment.py adds the user-requested3x2 view: loss, raw B/A,
raw/momentum adjacent cosine, raw/momentum norms. Raw cosine uses a trailing50
update mean; raw norms use a trailing10 update mean on logarithmic Y.

14_plot_loss_norms_by_lr.py adds an unsmoothed4x2 figure: all seven training-loss
curves first, then one raw-gradient/momentum norm panel per LR with common log Y
limits. It displays all712 accepted updates and marks the partial final batch;
the preceding figures and scientific inputs are preserved.

15_plot_raw_norm_final_to_minimum.py extracts final raw norm divided by its own
trajectory minimum over all712 updates and plots it against LR. The highest LR
has ratio77.60 versus5.90-11.29 for the others; the final partial batch is included
and labeled. Full-precision values and source/output hashes are retained.
