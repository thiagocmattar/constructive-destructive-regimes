# Compute and completion estimates

Select hardware from the exact workload: model, optimizer, sequence length,
microbatch, accumulation, precision, and diagnostics. Model parameter count alone
does not establish training memory fit. Historical timings from the other paper
are not current calibration.

Use local execution when it fits with headroom and acceptable runtime/impact.
Use a Runpod GPU Pod for sustained cloud training when the user has authorized
the spending envelope. Query price/capacity at launch time.

## User-owned DGX Spark

[Run020 calibration](../runs/020-2026-10-08-dgx-spark-calibration/REPORT.md),
2026-10-08, confirms SSH to `tcml-spark@spark-2277.local` and successful GB10
execution of all three sizes. The isolated user-owned ARM/CUDA13 PyTorch runtime
requires no Docker or sudo. At sequence 2048, effective batch 1024, microbatch 4,
FP16 AdamW with existing raw/adaptive FP64 diagnostics, measured post-warmup
updates take 25.14/28.68/38.27 seconds for 14M/31M/70M. One complete MiniPile pass
projects to approximately 5.0/5.7/7.6 hours per seed, plus unmeasured staging and
retrieval. Planning allowances are 5–6/5.7–7/7.6–9 hours.

Two post-warmup samples per size are a short calibration, not a sustained thermal
measurement or completion-time confidence interval. Peak allocated CUDA memory
is 5.45/6.37/8.52 GiB, with ample headroom. Fresh ARM initialization hashes differ
from cloud runs; load a common frozen checkpoint for exact matched starts. A new
scientific recipe, extra histories/diagnostics, or changed batch decomposition
needs its own calibration. The full training cache still needs staging. See the
report for phase accounting, source identity, verification and retained state.

[Analysis007](../analyses/007-2026-10-08-spark-runpod-timing/README.md) compares
completed matched RTX4090/RTX5090 runs with Spark: full passes took approximately
81/94/132 minutes on RTX4090 and 70/75–76/95 minutes on RTX5090 at 14M/31M/70M.
Spark's per-update times are 3.45–3.68× the RTX4090 and 4.32–4.80× the RTX5090. These historical
workload measurements exclude cloud provisioning/setup/transfers; they are not
a controlled isolated-GPU benchmark or a guarantee for a new recipe.

## Representative calibration

After warmup, measure multiple complete updates, including batch assembly,
host-to-device staging, forward/backward, diagnostics, and optimizer work. Record
sample count and variability. Separately measure full validation, setup,
checkpoint saves, final diagnostics, transfer, and local hash verification.

```text
ETC = provisioning + setup + steps * representative_update_seconds
      + validation_passes * representative_validation_seconds
      + diagnostics + checkpoints + transfer_and_verification
maximum_total_cost = maximum_hours * gpu_count * live_price_per_gpu_hour
                     + bounded_storage_and_other_cost
```

Count validation/checkpoint passes without double-counting a cadence boundary
that is also final. Include all workers, failures, and retries in the total budget.
For unequal workers, estimate per-worker durations/costs and their parallel
critical path separately; multiplying one ETC by GPU count is not a fleet model.

`tools/estimate_etc.py` performs the arithmetic for a single homogeneous worker
or synchronized group. Its p90 result applies empirical p90 phase samples; it is
a planning scenario, not a statistical p90 completion guarantee. Fixed overheads
and prices are supplied inputs, not live measurements made by the helper.

Before cloud launch, record a timestamped price, maximum duration/cost, hard
deadline with retrieval margin, storage persistence, required artifact inventory,
monitoring cadence, warning thresholds, and recovery/teardown plan. Retained
volumes keep billing; stopped compute is not complete resource cleanup.

During execution, refresh remaining-time estimates from completed work and account
for final validation, saving, and retrieval. Cost estimates remain estimates until
provider billing covers the full resource window.
