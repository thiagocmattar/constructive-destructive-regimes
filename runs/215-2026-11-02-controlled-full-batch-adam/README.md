# Run215: controlled full-batch Adam production

User approved Analysis036 calibration and requested full training on November2.
H027 / Analysis036. Nine independent1024-update arms (9,216 total updates):
MLP1e-4/3e-4/1e-3; ResNet32 and Transformer3e-4/1e-3/3e-3.
One frozen Analysis024 seed0 initialization per case, fresh Adam (0.9,0.999),
epsilon1e-8, constant LR; no decay, clipping, schedule, Hessian/HVP or extra seeds.
Data, reductions, physical partitions, parameter scope and precision match Run213.

Every optimizer boundary measures the complete raw gradient; all four optimizer
streams retain exact prefix diagnostics. Full validation/test at0,128,...,1024;
training loss at every boundary. Recovery every64 steps plus final1024 stores
model, Adam, RNG and diagnostic states. Raw audit gradients at0/1/1024 persist.

The approved eight-row figure layout uses training loss (log), eta*kappa_hat,
rho for raw/momentum/Adam, separate gradient/momentum norms, and R_t only for
Adam's actual updates. Training diagnostics remain unsmoothed; plotting uses
16-step trailing means, and R_t remains cumulative without smoothing.

Scientific inputs freeze before execution. Production restarts from initialization;
no calibration history is joined or silently resumed. Adam diagnostics were copied
unchanged into src/eos_tasks/adam_diagnostics.py because this is their second run;
Run213's executed sources are preserved. See execution_plan.json, input_manifest.json,
qualification receipts, detached controller/guard, and attempt-owned manifests.

The three local MLP production arms completed all1,024 updates and passed
05_verify_results.py, including full test coverage and checkpoint/stream audits.
Cloud training is active. The user explicitly requested continued capacity search
and parallel use of suitable non-RTX GPUs. 08_parallelize.py keeps searching while
new workers boot and qualify, assigns one arm per Pod, and records device per arm.
The first Transformer trajectory continues unchanged; only its waiting shell was
paused to remove its two future queued arms, with an independent terminal watcher.

Operational bounds in execution_plan.json supersede the initial infrastructure
budget note in the frozen config: main stage USD45, soft stop41, compute stop44,
8.5h from first cloud creation,20-minute retrieval margin. Calibration cost USD2.26
is additional but stays inside the approved total USD50 envelope. Each new Pod has
60GB container disk plus20GB task volume; no network volumes. Admission requires
measured projected arm cost <=USD10 for Transformer or <=USD6 for ResNet and the
frozen4h arm limit. Hardware variation is recorded and limits LR-only attribution.

Active control records/logs stay under ignored scratch/. The independent guard
survives controller failures; detached remote jobs survive SSH loss. Completed
workers are retrieved with archive and per-file size/SHA checks before deletion.
09_finalize.py runs complete-cohort verification after cloud closeout. Missing or
rejected arms remain explicitly incomplete and cannot pass its nine-arm check.

Launch verified at2026-11-02 13:58UTC (10:58BRT): all six cloud arms have accepted
production updates and passed per-worker source/start-gradient qualification.
See production-status.json for exact task/LR/device mapping, numerical errors,
steps, loss, token coverage, memory, live quotes and measured ETC. Three RTX4090s,
one RTXPRO4500SE, one RTXPRO4000 and one24GB PRO6000 partition are active, with
one arm per worker. Four unsuccessful infrastructure/qualification workers were
retired before any scientific trajectory. Qualification logs and retrieval
receipts are retained; pre-delivery failures have zero scientific files.

At this snapshot, the slowest observed remaining duration is2.88h. Extrapolating
each observed rate and allowing9 minutes retrieval per worker gives USD14.47
for the study including calibration. This is an early estimate, not posted
billing; the approved USD50 ceiling and independent limits remain in force.
The search also tried non-RTX L40S/A100/H100 capacity; suitable available workers
were accepted regardless of model. The final1,024-step histories remain pending.

Progress at2026-11-02 15:17UTC (12:17BRT): four of nine arms are complete and
verified (all three MLP arms and ResNet LR0.003). The completed cloud worker was
deleted after scientific and per-file integrity verification. Five arms remain
active without reported errors; the slowest measured ETC is about95 minutes,
plus retrieval. See progress-2026-11-02-1517UTC.json and results.json.
