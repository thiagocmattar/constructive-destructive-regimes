# Analysis 007: Spark versus historical Runpod timings

Status: complete; local operational comparison only. No training was launched.
The user asks whether the Spark is slower than the Runpod GPUs used in this
project. Source runs are 002–010 (completed RTX4090/RTX5090 full passes) and
020 (GB10 Spark calibration). No hypothesis or manuscript claim is added.

For one full MiniPile pass, one seed, effective batch 1024, sequence length 2048,
microbatch 4, and the same FP16 AdamW/raw-adaptive diagnostic recipe:

| Model | Spark projected duration | RTX4090 actual duration | RTX5090 actual duration | Spark/4090 update time | Spark/5090 update time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14M | 4 h 58 min | 1 h 21 min | 1 h 10 min | 3.68× | 4.32× |
| 31M | 5 h 40 min | 1 h 34 min | 1 h 15–16 min | 3.65× | 4.53× |
| 70M | 7 h 34 min | 2 h 12 min | 1 h 35 min | 3.45× | 4.80× |

Ratios compare mean complete-update durations, excluding the first warmup update
and final partial update from each cloud run. There are 710 cloud observations
per run, one RTX4090 seed and two RTX5090 seeds per size, versus two post-warmup
Spark samples per size. RTX5090 ratios average the two run ratios. Durations use
actual cloud manifest start/finish times and the Spark's measured phase-based
projection. Cloud duration includes preparation inside the training process,
validation and checkpoints; provisioning, environment setup, and transfers are
excluded. Spark's projection excludes preparation/staging/transfer and uses
proportional final-partial-update accounting. The point estimates have different
coverage and should not be treated as controlled hardware benchmarks.

All nine cloud runs cover the same full token budget and match the reference
training dictionary and source-cache hashes. Cloud seeds/order and hardware
vary together. Spark uses ARM/CUDA13; cloud uses x86/CUDA12.8, and initial tensor
hashes differ. The comparison describes observed complete systems on this
workload; it does not isolate GPU hardware causally or establish speed for a
different model/recipe. Extra temporal-sketch runs are excluded because their
diagnostic workload differs.

All three sizes sequentially total about 18.2 hours on Spark, 5.1 hours on RTX4090,
and 4.0 hours on RTX5090. Separate cloud workers can additionally execute sizes
in parallel; the 5090 cohort's training critical path is approximately 95 minutes,
with extra setup/retrieval time. Spark has ample capacity and no cloud GPU rental,
but completed jobs take materially longer. No current prices were queried or
cloud spending authorized by this comparison.

[00_compare.py](00_compare.py) verifies completed coverage/config matching,
reduces raw event durations, and records source byte sizes/SHA-256 identities in
[results.json](results.json). Original evidence is preserved. No figures are
needed for this three-row timing table.
