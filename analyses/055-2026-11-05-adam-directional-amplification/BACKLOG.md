# Backlog: preserve the useful ideas, keep the current task bounded

All items below are deferred, not launched or silently included in Analysis055.
No cloud spending, training recipe, model choice or new scientific budget is
authorized by this list. Preserve unfavorable existing results when revisiting.

- [ ] **TODO2: confirmatory forecasting contract.** Fix horizon, loss threshold,
  quiet eligibility, onset/escalation distinction, terminal censoring and the
  pre-update clock. The inherited8-update/10% marker used descriptively here is
  development evidence. An alarm at+0.4% can be useful; quantify lead to first
  increase, pronounced crossing and peak separately. Labels use future loss,
  never the diagnostic itself or a future peak-gradient feature.
- [ ] **TODO3: fair classifier/component study.** Estimate future-excursion
  probability from current/past state; compare loss/context, norm level/growth,
  rho, their joint representation and Q. Q is a deterministic combination,
  not an independent third signal. Match data, feature availability, capacity
  and tuning budget. Audit/reuse the archived models before replacement.
  Include ordinary periods and zero-event arms; inspect per-arm as well as
  pooled metrics. Do not optimize thresholds on inspected test blocks.
- [ ] **TODO4: untouched validation.** Freeze the method before new seeds or
  whole trajectories. Measure calibration, precision-recall, false alarms per
  interval, event recall at fixed alarm budget and practical lead times. State
  dependence precludes naive independent-step confidence intervals. Conditions
  for any new collection need a separate concrete design and bounded budget.
- [ ] **TODO5: action utility.** Compare the same action/duration triggered by
  geometry, norm and loss, plus unchanged and fixed-lower-LR controls. Monitor
  complete trajectories including false alarms, replacement spikes, validation,
  equal-update and wall-time progress. Reuse038-053: temporary gates and raw
  clipping have documented failures; input clipping does not bound Adam steps.
- [ ] **Beyond200M feasibility demonstration.** Select one appropriate >200M
  workload only after a concrete memory/time/cost plan. Benchmark observer off
  versus previous-gradient capture; record identical-update checks, memory,
  wall time, distributed reductions if relevant. Retain raw/clipped/adaptive
  distinctions. Computation without spikes shows feasibility, not warning value.
- [ ] **Compare with Bai's HVP predictor.** At a tractable case, distinguish
  their preconditioned-Hessian directional quantity from our mixed path response;
  compare cost and usefulness on identical decision states. No inferred equality
  or new high-dimensional Adam stability theorem.
- [ ] **Batching and stabilization mechanism.** Existing SGD attenuation motivates
  the idea but cannot establish Adam noise stabilization. A future matched Adam
  design must separate sampling effects on the measured rho from changes in full
  gradient/path geometry, and evaluate norms, loss and optimizer-state response.
  Reducing absolute negative rho raises the noiseless GD norm-growth boundary;
  it does not prove that sampling reduces gradient growth or future spikes.

The manuscript remains unchanged until the author has reviewed and understood
the evidence. No promise of a universal threshold, novel optimizer or superior
generalization is part of the present argument.
