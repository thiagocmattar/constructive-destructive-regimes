# Learning rate schedule intervention at the midpoint

Status: complete. All six branches were retrieved and independently verified;
their workers/volumes are removed. The user-selected single-seed first test uses
Pythia14M/70M, seed 1235, from Runs 005/007 update 356 checkpoints. Original decay,
held midpoint LR and twice-as-fast decay continue through update 712. Multiple
seeds are reserved for later validation. Each branch runs on a separate GPU for
minimum elapsed time, using the same RTX5090 software stack as its parent.

Raw-gradient capture, optimizer state, data order, accumulation and precision
match the parents. Initial/model/order identities are pinned. Restoration tests
include stochastic uninterrupted/resumed equivalence and actual parent states.
Before full execution, CUDA mechanics must closely match the archived original replay
and the common first gradient across schedules. Mechanics reset to the immutable
parent before each scientific continuation.

Compare raw loss and adjacent cosine with fixed windows and 31-update smoothing;
primary descriptive contrast uses pairs 358-712. Keep frequency F, unconditional
negative intensity D and conditional magnitude M separate. Validate all 500
documents and 693329 targets at 356/445/534/623/712, including the packed tail.
Whole-trajectory A/B/R includes the inherited prefix and is labeled accordingly.
Single-seed contrasts are exploratory, conditional on the trained prefix; the
intervention also changes LR-scaled AdamW shrinkage. No causal mediation,
convergence or cross-seed uncertainty claim follows from alignment sign.

Executed cloud envelope: six Secure RTX5090 workers at checked USD 0.99/GPU-hour,
maximum two hours per worker and USD 15 aggregate including running storage and
bounded retries.60GB container/40GB persistent storage per worker; no network
volume. Expected cloud completion1-1.5hours and USD 7-10, qualified from measured
parent updates plus setup, validation, recovery and verified retrieval. Capacity
is LOW; the estimate assumes six workers can be allocated. No payment or cloud
spending is implied for the deferred seed validation cohort.

Use detached jobs, a20-minute retrieval margin and local stop-at-deadline guards.
Read-only monitoring at bounded intervals; preserve unrelated resources. Retrieve
and verify every artifact size/SHA and independent scientific checks before
removing each worker and its task storage. Manifests, progress logs and new
recovery states at 534/712 survive dropped SSH connections.

Preparation:01_prepare.py; cloud operations:02_operate.py; prelaunch CUDA
qualification:03_qualify.py; independent retrieval acceptance:04_accept.py;
reductions/figures:05_analyze.py. The run configurations and source inventories are
frozen in each owning run. The completed pilot and replication analyses remain
unchanged and are inputs, not overwritten by this intervention.

## Result and closeout

Accelerated decay reduces negative frequency/intensity and brings the declared
positive-smoothed criterion 65/101 updates earlier at 14M/70M. Held LR keeps
alignment mostly negative and reaches the lowest validation loss at both sizes;
accelerated decay reaches the highest. Both decaying70M branches later return
to negative smoothed alignment. This supports schedule-sensitive raw-gradient
cancellation, without identifying alignment sign as convergence or learning quality.
F/D size ordering reverses under held LR in this single-seed pilot. The earlier
three-seed original-recipe conclusions are preserved; extra seeds are deferred.

See the [conclusion](REPORT.md), [full observation](observations/001-schedule-intervention.md),
[results](results.json), and [curve data](curve_data.csv).
Publication figures: [14M loss/cosine](figures/loss_and_raw_cosine_14m.pdf),
[70M loss/cosine](figures/loss_and_raw_cosine_70m.pdf), and
[schedule/validation](figures/schedule_and_validation.pdf).
All three pages passed rendering/visual review and text-bound checks.

Cloud elapsed 87.91 minutes; estimated compute/storage USD 6.92, below the USD 15
cap. All six deletions and guard cancellations are recorded; zero task cloud
storage is retained. Billing is provisional: the latest filtered API snapshot
shows USD 2.36 in incomplete hourly records and does not cover the full window.
[Cloud closeout](cloud_closeout.json) and [billing snapshot](billing_snapshot.json)
retain the receipts. Unrelated cloud experiments and all source-run/manuscript
evidence were preserved. No additional seeds were launched and no manuscript asset was edited.

## Numerical qualification and infrastructure observations

The first restored 14M gradient matches the archived parent (loss exactly;
raw-norm difference3.25e-8, cosine difference7.66e-9). The initial strict replay
gate failed on later cosines: maximum difference1.98e-4 over three updates,
with maximum loss difference4.79e-6. An unchanged-input repeat is retained in
Run 011's attempt alongside the failed-gate record. GPU replay is numerically
close, not bitwise equivalent. Mechanics acceptance was revised before the
control's full continuation to absolute cosine tolerance1e-3, loss1e-4 with
relative1e-5, raw norm relative1e-3/absolute1e-6, and ratio relative1e-5/absolute1e-6.
First-gradient checks use relative1e-5/absolute1e-7 against the parent and
between branches. An initial tighter cross-branch relative1e-6 gate rejected
Run 016's first cosine difference1.37e-7 from Run 014 (identical first loss;
raw-norm difference5.96e-8); it was aligned with the existing parent tolerance
before Run 016's full launch. No scientific training input was changed.
Record this qualification revision and numerical variation in interpretation;
use the newly executed control for schedule contrasts, retaining the archived
parent replay comparison as a sensitivity check.

One preflight checkpoint SCP disconnected (Run 014); its unchanged compressed
packet was retransferred and size/SHA checked. Run 012's interactive extraction
query timed out, but its parent hashes passed the loader before mechanics.
The late-created lease files were separately uploaded before full launch.
None of these operational repairs changed checkpoint, token or schedule bytes.
All six pods are task-attributed, separately leased; no unrelated resource is managed.
