# Execution feasibility for the next intervention study

Read-only audit, 3 November 2026. No training, calibration, provisioning, account change, global tool update, or cleanup was performed. The current account resources were preserved.

## Working account access and current catalog

No Runpod MCP tool is exposed in this tool session. Plugin Management search for Runpod returned an empty result; this is not proof that the installed skill bundle is absent or that the service cannot connect. The installed portable CLI works with existing authentication:

`C:/Users/thima/Documents/Doutorado/exps/constructive-destructive-regimes/tools/bin/runpodctl.exe`

Version is `2.14.0-dd55bcf`. Its verified read commands are `user`, `gpu list --include-unavailable`, and `pod list --all`. Default sandbox access failed DNS resolution; the same read-only commands succeeded with sandbox escalation. No credentials were printed or copied. No global tool update was attempted; a deployment should refresh/pin the portable tool as its own preparation step.

At 14:59:09 UTC / 11:59:09 Sao Paulo, the API reported **USD23.6126894701** balance and **USD0.016/hour** current account spend. Two existing Pods report stopped/EXITED. The Pod `costPerHr` field still shows USD1.78; it should not be confused with the current account burn. Neither resource was attributed to this task or changed.

Live Secure catalog prices and capacity at that snapshot:

| GPU | USD/hour | Live capacity |
| --- | ---: | --- |
| RTX4090 | 0.89 | None |
| RTX5090 | 1.19 | None |
| RTX6000 Ada | 0.99 | None |
| A100 PCIe / SXM80 | 1.79 | None |
| A40 48GB | 0.59 | Low, EU-SE-1 |
| RTX PRO4500 Blackwell Server Edition 32GB | 0.72 | Low, US-KS-2 |
| RTX2000 Ada 16GB | 0.24 | Low, EU-RO-1 |
| H100 SXM80 | 3.99 | Low, EUR-NO-2 / US-MO-1 |
| B300 | 8.99 | Low, EUR-IS-1 |

These are catalog observations, not a capacity reservation. The public official [pricing page](https://www.runpod.io/pricing), updated 28 October, corroborates the standard quoted 4090/5090/A100/6000Ada rates. Requery at launch. A40 and PRO4500 need workload qualification; the cheapest hourly rate need not be the cheapest completed experiment. The expensive H100/B300 options are poor default choices for a USD20 stage without a measured speed advantage.

`read_only_preflight.json` retains the redacted account/catalog snapshot. The user's USD23 is confirmed approximately; additional deposits of USD50/100 imply balances of about USD73.61/123.61, not total budgets of USD50/100. Suggested task ceilings below leave reserves and do not authorize those extra deposits.

## Measured historical workload

All timings below were reduced from Run215's actual `artifacts/attempt-01/*/lr-*/metrics.jsonl` field `seconds` and `manifest.json` fields `elapsed_seconds`, `completed_updates`, and `environment.gpu`. `historical_timing.json` records every retained arm. The per-update measurements include diagnostics, with periodic evaluation spikes; total arm elapsed also includes checkpoints and final work.

| Case / historical device | Median update | 90th percentile | 1,024-update full arm |
| --- | ---: | ---: | ---: |
| MLP LR0.001 / owned RTX5070Ti Laptop | 0.01168s | 0.01294s | 14.90s |
| ResNet LR0.003 / RTX4090 | 2.60465s | 2.60864s | 2,674.24s / 44.57min |
| Transformer LR0.003 / RTX4090 | 9.21412s | 9.32084s | 9,429.61s / 2.62h |
| ResNet LR0.0003 / RTX PRO4000 | 6.94657s | 6.96726s | 7,217.71s / 2.01h |
| Transformer LR0.001 / RTX PRO4500 Server | 10.53438s | 10.62363s | 10,879.94s / 3.02h |
| ResNet LR0.001 / PRO6000 MIG24 | 9.62027s | 9.63820s | 9,836.21s / 2.73h |

These are historical device/recipe/state measurements, not today's measured ETC for modified methods. ResNet LR and device co-vary across these arms. The common full objectives are 5,000 MLP images, all50,000 ResNet images in fixed ghost-BN groups, and 2,088,600 Transformer supervised targets per update. Physical Transformer chunk64 is the frozen Run215 contract; changing to the faster chunk512 requires a new frozen run and within-comparison qualification.

The MLP is extraordinarily inexpensive on the owned GPU and can support rich development, negative controls and independent seeds without cloud charges. Implementation and checks, not MLP arithmetic, dominate its calendar time. This removes any need to spend the first cloud dollars on debugging optimizer-state rollback or elementary secant formulas.

## Replay and storage burden

Run215 `01_train.py` overwrites `recovery.pt` at every save and renames it `final.pt` at completion. Its scalar histories do not recover weights/moments at the proposed forks. New matched common prefixes are required: MLP657, ResNet383 and Transformer755 updates, 1,795 total. Save immutable fork snapshots before consuming the fork gradient, preserving model, Adam moments/ages, RNG, data/BN conventions and diagnostic history. Qualify the historical event and prove that each untreated branch replays the newly generated control. A failed historical replay does not invalidate a fresh matched study, but event reselection must precede treatment inspection and be declared.

Existing complete checkpoint sizes are approximately 55.19MB MLP,39.12MB ResNet and1.166GB Transformer. These include large full-stream diagnostic buffers. Analysis038's conservative eight-arm plan retains35 files per case, roughly44.1GB altogether. With nine arms and the same snapshot policy it becomes39 files per case, about49.1GB before inputs and extra audits. The shared Run215 input inventory is1.087GB. A purpose-designed lean recovery format could lower storage substantially, but must preserve all required method/diagnostic state; measure before quoting a new allowance.

Do not retain dense gradient histories throughout every arm by default. A Transformer raw FP32 gradient is roughly59.3MB: a257-gradient window is15.2GB;1,025 gradients are60.8GB per arm, before extra streams. Multi-lag geometry needs a planned selected-window/ring-buffer/cadence protocol. Scalar norms/cosines alone cannot reconstruct new cross-lag products afterward.

## Cost arithmetic for an exploratory pilot

Let `U_case` be a calibrated complete-update time and `F_case` a calibrated full-objective forward time. For `a` arms of128 updates per case:

```text
MLP updates = 657 + 128*a
ResNet updates = 383 + 128*a
Transformer updates = 755 + 128*a
ordinary GPU seconds = sum_case(updates_case * U_case)
Armijo trial seconds <= sum_case(128 * 8 * F_case)
total cost = sum_workers(hourly_price * billed_seconds/3600)
             + storage + setup/qualification/retries/retrieval
```

The exact objective forward is not measured separately in the records inspected here. Adam's Armijo branch adds up to1,024 trial forwards per model, not extra backwards. Moment buffers and the optimizer clock advance once per accepted update; all rejected candidates roll back. The max-trial cap can stall. A secant-only candidate should add vector reductions rather than extra objective passes; a backtracking version must charge its trials explicitly.

Using only the historical RTX4090 median times as a scenario:

| Three-model exploratory design | Nominal total updates | Paid ResNet+Transformer ordinary GPU-hours | Ordinary compute at USD0.89/h |
| --- | ---: | ---: | ---: |
| Original eight arms | 4,867 | 5.57 | 4.96 |
| Eight arms plus one secant candidate | 5,251 | 5.99 | 5.33 |

MLP executes locally in these calculations. The deliberately conservative scenario `F_case <= U_case` adds up to3.36 paid GPU-hours /USD2.99 for the two Armijo branches; this is an assumption for planning, not a measured bound. Thus ordinary-plus-trial arithmetic is aboutUSD7.95/8.32 before setup, qualification, new diagnostic overhead, storage, retries and transfers. A1/A2/A3/A4/A5/A6 and a no-trial secant method can have different objective trajectories and wall times. Qualification and method-specific timing must replace this scenario before launch.

With the historical Transformer PRO4500 median10.53s, its nine-arm ordinary segment is5.58h atUSD0.72/h, aboutUSD4.02 before trials and setup. ResNet on that exact GPU has no inspected matched timing. Do not port its PRO4000 rate as a measurement on the PRO4500. A qualified A40/4500 can be economically attractive, but may exceed the initial time scenario.

## Bounded stages

**Stage A: existing USD23.61; task ceiling USD20; reserve USD3.61.**

1. Develop, test and qualify all proposed mechanics locally on MLP. Run the exploratory MLP comparison under an explicit selected design; cloud charge zero. Choose whether to retain the original eight arms, add a secant arm, or narrow. Fix recovery and long-lag capture now.
2. Allocate at mostUSD2 for paid readiness/qualification, then at mostUSD13 for the qualified ResNet/Transformer exploratory comparison, withUSD5 reserved for transfer, storage, retries and contingency. These are allocations under oneUSD20 aggregate task limit, not three independent spending permissions.
3. Use at most two paid workers concurrently and a12-hour task wall deadline, with an independent aggregate-cost guard. For a price ceilingUSD1.19/hour, start retrieval/termination before accumulated active spend reachesUSD16, reserving four dollars for up to two workers' retrieval time and bounded storage. A resource timeout is not an assurance of backend deletion; the guard needs actual stop/delete verification. Requery stock and price at allocation. A slow placement that cannot fit the remaining envelope is not a reason to exceed it.
4. This stage buys selected-event action/mechanism evidence and implementation qualification, not a final independent-seed online efficacy claim. Preserve late/replacement spikes, failures and early stops. Additional methods and all-case confirmation wait for the result.

**Stage B: additional USD50; balance aboutUSD73.61; proposed cumulative task ceilingUSD65, reserve aboutUSD8.61.** The extra stage allowance isUSD45 after the initialUSD20. Use it for one predeclared confirmation question, not a grid of every method. A three-seed, four-policy,1,024-update full-batch confirmation on one model is12,288 updates. At historical RTX4090 rates, ResNet is8.89 GPU-hours/ USD7.91 ordinary compute; Transformer is31.45 GPU-hours/USD27.99. Setup, data, saving, new diagnostics and repeated evaluations are additional. If Armijo is one of the four policies, three1,024-step arms may require up to24,576 extra forwards, materially increasing the cost. Keep it in short windows as a safeguard reference unless its full confirmation is explicitly budgeted. A qualified no-trial four-policy Transformer comparison can plausibly target aUSD40–45 ceiling, but only a calibrated launch packet can establish that it fits. With three workers, allow up to24 hours wall time and guard the extraUSD45; a holdout seed should never be substituted after inspection.

**Stage C: additional USD100; balance aboutUSD123.61; proposed cumulative task ceilingUSD110, reserve aboutUSD13.61.** After the initialUSD20,USD90 can support one independent mechanistic confirmation and one practical stochastic AdamW bridge, conditional on measured runtimes. Existing full-pass timings imply three seeds x four policies on Pythia70M costs26.4 RTX4090 GPU-hours/USD23.50 ordinary compute (12 x132-minute historical passes), or19 RTX5090 GPU-hours/USD22.61 (12 x95-minute passes). These are recipe/hardware historical scenarios, excluding setup, new capture, validation and checkpoint transfers; choose aUSD35–45 subcap after calibration. Initialization and data-order seeds, common tokens and schedule clocks must remain paired. Additional budget should purchase independent information and a properly matched baseline, not a larger single model by default. A one-model practical bridge plus one full-batch confirmation may fitUSD90 additional, but the exact plan is conditional and not a promise.

No stage claims that spending a specified amount guarantees novelty or acceptance. At each gate, only the next bounded stage is released. Preserve the current unrelatedUSD0.016/h spend and stopped resources; task guards must monitor task-owned resource IDs and not terminate the account wholesale.

## Reusable code and concrete preflight work

- `src/eos_tasks/production.py`: full-objective `ProductionTask.gradient/evaluate`, fixed reductions and BN groups. Reuse unchanged where possible.
- `src/eos_tasks/task.py`: pinned model construction and numerical configuration.
- `src/eos_tasks/adam_diagnostics.py`: current four-stream observer; keep raw/input/moment/adaptive/applied step distinct. Its `consume` is called after the outgoing update. An online secant candidate needs a pre-update observation phase using the preceding gradient and preceding actual step, not an outgoing product mistaken for an available predictor.
- Run215 `01_train.py`/`02_qualify.py`: frozen-input checks, initial gradient qualification, full split coverage, boundary clocks and arm manifests. Add interventions in a new run-local driver; do not mutate executed Run215 inputs. Save immutable fork/milestone states plus a separate rolling crash-recovery file.
- Run215 `03_cloud.py`/`08_parallelize.py`: provenance for detached workers, exact staging, budget guards and parallel recovery. Extract only necessary deployment logic and requery resource/catalog identifiers; do not replay their old IDs or deadlines.
- Add exact cross-step products `s_previous dot (g_current-g_previous)`, `s_previous dot g_previous`, `||s_previous||`, and incoming/retained optimizer-state quantities before applying the current action. Keep the availability clock separate from post-update logging.
- Test independent Adam moment recurrences; history-aware beta2 bias correction; clipping after full accumulation; actual-step rounding; first-order direction fallback; finite-trial rollback; RNG/BN restoration; complete reproducible resume; immutable fork hashes; zero norms/nonfinite cases; trigger timing and diagnostics-without-update-change.
- Benchmark ordinary forward/backward, pure forward, validation, snapshot save/reload and artifact transfer separately. Observe real GPU peak allocations with selected temporal buffers. Qualify any new physical partition before frozen production.
- Before billable execution: exact scientific design/configs, initialization/seed coverage, update/token budget, purpose of each control, live price/stock, calibrated ETC, aggregate-cost guard, wall deadline, storage ceiling, raw inventory/retrieval checks and task-owned teardown verification.

The strongest practical next step is a rich owned-GPU MLP development study, followed by a cloud exploratory study that already captures the exact products needed for the final question. This is compatible with ambition: old evidence guides good measurement, while new experiments supply the missing independent and actionable comparisons.
