# Analysis023: smallest-LR, 3,000-update Spark calibration

**Latest instruction:** finish the active Run075 and the two agreed3k
calibrations, retrieve/analyze the results, then wait for user review. Do not
start any subsequent training. The finish-only override below supersedes the
original predecessor-resume plan; scientific inputs remain unchanged.

The user selected a first run at the smallest LR for each case, with3,000
full-GD updates, on owned Spark. The earlier20k multi-seed expansion and
Runpod spending proposal are not being launched. On October15 the user also
asked to focus on runs/analyses and defer draft/restatement work: this pipeline
does not modify, insert into or compile any manuscript.

## Comparison

| Dataset / model | LR | Seed | Horizon | Evidence |
| --- | ---: | ---: | ---: | --- |
| CIFAR-10 first5k / tanh200-200 MLP | 0.004 | 0 | 3,000 | Reuse verified Run059 prefix |
| CIFAR-10 all50k / ResNet-32 | 0.1 | 0 | 3,000 | New Run120 |
| WikiText-2 / two-layer Transformer | 0.05 | 0 | 3,000 | New Run121 |

This calibrates runtime, transition coverage and late diagnostic behavior before
committing to a wider LR/seed grid. It does not assume that all three smallest
LRs must exhibit a reversal, reach an EOS regime, or converge within3k. One seed
and one LR per case provide exploratory evidence, not replication or an LR
effect estimate. Dataset, model and objective differ across columns.

Use Analysis020's frozen inputs and qualified scientific code, exact seed0
initial tensors, full objective/physical decomposition, FP32 training with TF32
off and FP64 geometry. Plain GD uses no momentum, decay, clipping, scheduler,
augmentation or dropout. The new configs change LR/horizon and retained states
from their500-step controls, with new run IDs. Old executed configs are intact.

One iteration is one full-objective parameter update:5k MLP images;50k ResNet
images averaged over the same50 fixed1k ghost-BN groups; or59,680 Transformer
sequences /2,088,600 supervised target tokens. Complete prepared losses retain
the documented token weighting and exclusions. No objective is shortened for
speed. The cached data, initial tensor and source hashes are checked by the
unchanged training runner and frozen packet inventory.

## Diagnostics and retention

Every update records unsmoothed objective, raw norm and adjacent cosine,
q=eta*kappa_hat=1-rho*norm(g_t)/norm(g_previous), A=sum(norm(g)),
B=norm(sum(g)), B/A, energy, exact all-pair and lag-one/beyond-lag-one terms,
actual parameter displacement and numerical update residuals. First adjacent
pair/zero norm is undefined; no denominator epsilon is inserted. This q is a
projected finite-step gradient change; local Hessian curvature interpretation
requires a Taylor approximation. Its relation to rho is algebraic.

Largest algebraic Hessian eigenvalue: states0,100,...,3000, fresh relative
residual<=0.005, same solver and declared fallback as Analysis020. Hessian
coverage is full5k for MLP, first5k ResNet images, first2,500 Transformer
sequences. The latter two are explicitly subset spectra. This calibration does
not add an unbudgeted full-Hessian study. Reused Run059 used the two-eigenvalue
solver; select its largest value and disclose the diagnostic implementation
difference. Gradient norm at the final recorded update is at state2999;
final loss and spectrum are at state3000.

Rolling recovery every100; retained states500/1000/2000/3000 plus final for the
new runs. Recovery includes model, optimizer, RNGs, previous gradient and prefix
geometry; no dense gradient history. Full test evaluation every500 and final,
using the complete prepared test split and its original loss weighting. No test
metric selects learning rates. Reuse the first3000 Run059 rows, its next-row
loss at state3000, and31 spectral states; never use its20k endpoint as a3k value.

## Priority scheduling and verification

At launch the existing Run075 Transformer seed1 was active. The new detached
scheduler stops **only the old queue's admission process**, while its detached
training child finishes normally. It verifies the parent's command/path,
waits for the child to exit, verifies its terminal manifest and records its
transfer inventory. It then replaces only the scheduler and executes120 then121
sequentially. No active optimizer is paused and no jobs share GPU compute.

The original plan would have resumed unstarted Analysis020 conditions after
calibration. The user explicitly revoked that continuation on October15.
The two old admission processes were terminated while their detached active
training child continued normally. A verified finish-only supervisor now
retains that child's terminal evidence and admits **only120 and121**. It has
no path to execute the remaining020 conditions. Analysis020 Runs076-103 and
Analysis021's extra-LR sweep remain held until the user reviews the results.
The drain retains an8h bound; failure leaves subsequent admissions held.
No executed condition is replayed, and no scientific config/horizon changes.

The CPU-only Linux scheduling check passed before touching production:
active child finishes; next admission is blocked; child/inventory survive;
jobs execute sequentially; completed attempts are skipped on continuation.
The source/config packet is SHA256 checked locally and remotely. The existing
Analysis020 mathematical and full-shape smoke qualification is reused only
after verifying unchanged training code. Scheduling checks are mechanics,
not scientific results.

Remote priority scheduler PID468139 was launched at11:36UTC October15.
Initial verification found old admission PID426713 stopped, its active child
still running, and the priority queue draining Run075. This is a verified
waiting launch, not a claim that Run120 has already updated parameters.
The later finish-only override replaced admission PIDs468139/426713 with
supervisor474503. Live checks confirm both former processes are gone, active
child456314 continues, and automatic predecessor resume is disabled. CPU-only
success/failure checks verify that subsequent training stays held in both cases.
See `launch.json`, `scheduling_verification.json`, `initial_status_verification.json`
plus `finish_only_policy.json`, `finish_only_verification.json`,
`finish_only_live_verification.json` and the current attempt/queue manifests.
A detached local controller polls
every30min, retrieves terminal evidence, checks file sizes/SHA256, and runs
numeric analysis automatically. No manuscript action or visual acceptance is
automatic. Warning/failure states remain visible instead of dropping conditions.
On October15 the user requested fewer checks given the long ETC. Both local
monitors were changed from5min to30min and their Windows SSH polling children
use `CREATE_NO_WINDOW`. Only the local monitors restart; Spark training and the
finish-only policy remain unchanged. Detection of completion can lag by30min.

## ETC and storage

Complete500-update Spark runs project roughly12h for ResNet3k and26.7h for
Transformer3k: **38.7h additional compute**,54.2h planning allowance,60h hard
compute bound. These are qualified extrapolations: changed LR and late
eigensolver behavior are unmeasured. Add the active Run075's remaining roughly2h
at launch. The expected calibration finish is therefore about41h from launch,
excluding pending020 jobs, which now stay held after calibration. The MLP prefix is
already complete. This is owned compute with no cloud rental or storage bill.

Allow12GB additional retained evidence on both Spark and local disk; preflight
found3.7TB remote and219GB local free. Transfer archives are removed only after
the inventory/byte/hash checks pass; raw source/checkpoints remain on both
machines. At completion report actual elapsed time, coverage, memory and
transition/late-window summaries before choosing any larger grid or horizon.

Scripts: `00_prepare.py` freezes the two new configs and source packet;
`01_queue.py` retains the qualified original scheduler and execution helpers;
`02_launch.py` uploads, runs the CPU scheduling check and detaches;
`03_verify_scheduling.py` tests admission/continuation on CPU fixtures;
`04_collect.py` retrieves verified evidence; `05_reduce.py` verifies all used
scalar identities and summaries; `06_controller.py` monitors/retrieves/reduces.
`07_finish_only.py` is the active infrastructure override; `08_apply_finish_only.py`
installs it without interrupting training; `09_verify_finish_only.py` checks
its success/failure paths. The original deployed source packet is preserved.

## Available result

The requested [two-column3k overview](figures/eos_gd_overview_completed_3k.pdf)
now covers the completed MLP and ResNet cases, with unsmoothed loss,
eta*lambda_max, eta*kappa_hat and raw-gradient norms on logarithmic Y axes.
Run120 is retrieved/hash verified; all5,998 used pairs,62 spectra and6,000
raw norms pass checks, and the current eight-panel PDF passes
visual review. See [observation003](observations/003-completed-3k-overview.md),
`10_plot_completed_3k.py`, `completed_3k_results.json` and its figure manifest.
The Transformer continues separately; no manuscript or reference-PDF change.

The verified3k MLP prefix has loss0.596781 ->0.336508, late1k mean cosine
0.999999957, B/A0.794144, and maximum sampled eta*lambda_max0.316827.
No adjacent reversal occurs in this window. See
[the local observation](observations/001-mlp-3k-prefix.md) and
[machine-readable checks/results](mlp_prefix_results.json).
