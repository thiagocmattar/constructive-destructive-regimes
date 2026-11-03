# Observation001: fixed-step design, capacity and qualification

User requested overnight literature-motivated interventions on the idle Spark
for all three highest-LR model/dataset cases, with before/after graphs. Initial
planning512+512 was superseded before production after full-objective timings:
about0.017/15/35 seconds per MLP/ResNet/Transformer update. Final design uses exact
warm state1024 parents and64 new updates, ending1088. Ten primary conditions
plus two optional time-admitted cosine durations. H029; Run221 qualifications,
Run222 production; Analysis043 reductions. Primary control is unchanged Adam on
Spark, not the historical prefix alone.

Sources: Run221/PROTOCOL.md, runtime_profile.json, warm_parent_qualification.json,
parent_manifest.json and frozen input_manifest.json; Run222/config.json,
input_manifest.json, launch_receipt.json, and the verified raw snapshot referenced
by retrieval.json. Prepared model/data and complete original objectives remain
byte-pinned. The first profile fails before an update on metadata list indexing;
the corrected profile executes9 mechanics updates. Zero-update warm transfer
loads exact model/moment tensors for all cases. ResNet's raw gradient differs
0.00871875 relatively across GPUs, beyond the0.001 gate, with loss relative
error2.78e-7. That failed gate is preserved in Run221. Run222 explicitly amends
the question to matched Spark responses; it does not weaken the historical gate
or claim historical trajectory replay. MLP/Transformer pass that transfer gate.

All three Run222 cases pass exact same-parent Spark loss/norm/rho and2 direct
versus1+1 resumed weights/moments/previous raw gradient/RNG. Fifteen focused CPU
checks cover schedule endpoints and native warm AdamW parity. Frozen scientific
inputs remain unchanged after execution. The source Manifest pins the data,
model/task helpers, compact parents and their historical curves. No cold moment
reset or coupled L2-gradient penalty is introduced.

Figure contract: each intervention figure has three model/dataset columns,
with unchanged gray and intervention blue, solid unsmoothed trajectories and
independent Y ranges within each metric/model panel. Rows: task loss, raw
$\rho_t$, raw gradient norm, $\eta\widehat\kappa_t$, applied LR/base. At relative
update0 (state1024), a marked hardware/intervention boundary separates the shared
historical960..1023 prefix and matched Spark1024..1088 suffixes. Both curves
within a panel share units and scale; no pooled Y limit across model columns.
No interpolation of missing measured points or silent curve clipping. The
terminal LR drawing value only closes its last applied interval. Additional
parameter/actual/decay increment measurements remain available in source tables.

01_report.py reads only checksum-verified snapshots. Every plotted array is
asserted equal to the selected raw observations; comparisons use64 post-update
losses and the shared Spark fork loss. Endpoints use complete prepared held-out
splits for description, not tuning. Raw gradients, clipped optimizer inputs,
first moments and actual increments are separately labelled. Norm/rho alone
are not loss-change, curvature or stability guarantees under Adam.

This observation records design and qualification, not finished treatment
findings. Long-term convergence, method optimality, independent-seed robustness
and generalization are outside this one-state64-update comparison. Compute stops
by2026-11-04 07:45 BRT; report target08:00 BRT. Owned Spark, zero cloud spending.
Detached worker PID2439138, task-owned isolated directory; old queue untouched.
Active thread follow-up collects results, verifies checksums, reviews figures
and consolidates5-10 findings on completion. No manuscript edit or push.
