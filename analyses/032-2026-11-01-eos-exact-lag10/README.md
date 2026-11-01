# Analysis032: exact gradient cosine through lag 10

Completed: 27 short full-GD profiles from the nine Analysis024 conditions,
with all 270 cosines independently verified. There are 25 primary profiles;
two unqualified historical MLP reconstructions are shown separately.

- [Report and numerical endpoints](REPORT.md)
- [Primary figure](figures/exact_gradient_lag10.pdf)
- [All profiles, with reconstruction flags](figures/all_profiles_with_reconstruction_flags.pdf)
- [All 270 values](lag_profiles.csv), [results and sources](results.json)
- [Observation and captions](observations/001-exact-lag10.md)
- [Numerical verification](verification.json), [figure review](figure_review.json)
- [Execution closeout](closeout.json), [summary and timings](summary.json)

The windows are model states 500-510, 2000-2010 and 3000-3010. The last is a
separate continuation beyond Analysis024's 3000-state horizon. For anchor s
and k=1,...,10, the value is cos(grad L(theta_s), grad L(theta_(s+k))). Each
lag has one anchored pair, without averaging, smoothing, centering or projection.
A window contains 11 complete-objective gradients and ten original-rate GD
updates. Whole-model gradients and parameters stay FP32; diagnostic dot products
and norms use FP64. TF32 is disabled. Zero norms give explicitly undefined values.

| Case | Learning rates | Parent runs |
| --- | --- | --- |
| CIFAR-10 / tanh MLP | 0.004, 0.02, 0.04 | 059, 062, 122 |
| CIFAR-10 / ResNet-32 | 0.025, 0.05, 0.1 | 123, 124, 120 |
| WikiText-2 / Transformer | 0.05, 0.1, 0.2 | 121, 125, 126 |

Within each case, seed-0 initialization, data/order, objective and accumulation
are matched to the parent comparison. MLP covers the first 5,000 images with
half squared error summed over outputs and averaged over images. ResNet covers
all 50,000 images with the same 50 fixed 1,000-image train-mode BN groups.
Transformer covers all 59,680 fixed sequences and 2,088,600 target tokens,
using token-weighted NLL. No dropout, augmentation, momentum, clipping, decay,
schedule, Hessian measurement or held-out evaluation is introduced.

Seven conditions have exact saved states at all anchors: 21 saved starts.
Minimal copies preserve every model tensor, optimizer state and global RNG;
original and extracted checkpoint hashes establish provenance. Runs059/062
retain only initialization and their 20,000-update endpoint, so six requested
starts require replay. Replay losses/norms and available adjacent cosines are
compared with their original scalar records. Run059 qualifies at all anchors;
Run062 qualifies only at 500. Failed reconstructions at 2000/3000 remain in
the flagged companion and CSV, excluded from the primary figure. Scalar
qualification does not prove exact tensor recovery. Saved-state cloud results
are new continuations, with historical scalar agreement reported separately.

[Run205](../../runs/205-2026-11-01-eos-exact-lag10/README.md) owns the frozen
config, scientific code, inputs manifest and ignored raw evidence under
`artifacts/attempt-01/workerN/`. Its `source_inputs.json` and
`source_cifar_inputs.json` are byte-identical copies of the pinned source
protocol metadata. `checkpoint_provenance.json` records all 21 saved starts;
`packet.json` hashes the input/source packet. No original run or manuscript
asset was edited.

Three Secure RTX 4090 workers used the live catalog price USD 0.89/GPU-hour.
The preparation bound was two hours / USD 6 aggregate, with 30 GB container
disk and a 15 GB pod volume per worker, and no network volume. Detached jobs
persisted progress, full gradients and start/end recovery states; a separate
local guard bounded compute. Monitoring used approximately one-minute reads.
All 147 files (7.072 GB) passed local size/SHA-256 checks before task-pod
deletion. No task cloud storage remains. Compute cost is approximately
USD 1.05, provisional; the task-scoped billing reads have not posted yet.
Spark and unrelated resources were untouched.

`00_prepare.py` verifies original inputs/inventories and makes the packet;
`01_cloud.py` controls task-owned resources; `02_analyze.py` independently
checks vectors, reduces values and makes the plots; `03_record.py` records
observations and closeout. Scientific code and mechanics tests are run-local.
One initialization and one pair per lag provide no seed uncertainty or
stability/generalization test. Complete norm-weighted temporal accounting
cannot be inferred from these three anchored profiles.

Reproduce the local analysis with `02_analyze.py` and `03_record.py` using the
retained Run205 evidence. Cloud controls are a completed task record; a new
measurement needs a new run ID rather than rerunning the old allocation.
