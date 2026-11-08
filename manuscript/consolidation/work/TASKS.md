# Tasks for the next consolidation iterations

**All cards are proposed, not launched.** These IDs are local planning IDs.
Read [PLAN.md](../PLAN.md) before assigning a card. Each card specifies the
permitted output and acceptance criteria; it does not authorize unrelated
manuscript edits, new science, infrastructure, or publishing.

## Queue and dependencies

| ID | Task | Type / readiness | Dependencies |
|---|---|---|---|
| P01 | Closest-method and novelty comparison | Literature/planning; ready to assign | Author accepts the working contribution question |
| P02 | Mathematics, indexing, and information timeline | Retained-source audit; ready to assign | None beyond this plan |
| P03 | Exact evidence/comparator inventory | Retained-data inventory; ready to assign | P02 for time labels |
| P04 | High-LR event characterization table | Retained-data analysis; ready after its event contract is selected | P02/P03 |
| P05 | Publication storyboard and compact panels | Replot existing evidence; wait for storyboard agreement | P02-P04 |
| E01 | Same-fork intervention comparison and replication | New experiments; design/cost approval missing | P01-P03, author chooses C3 scope |
| E02 | Isolated runtime/memory/overhead | New benchmark; execution scope missing | E01 policy/probe contract |
| E03 | Probe robustness | Retained replay/new capture; resource plan missing | P03, precise observer contract |
| E04 | Ordinary minibatch transfer | Conditional new experiment | Required only for a practical stochastic-training claim |
| E05 | Quiet/spike spectra and local-subspace tests | Conditional retained-data/replay analysis | Required only for spike-specific rank/stationary-axis claims |
| E06 | Prospective spike forecasting | Conditional separate study | Required only for a forecasting contribution |
| W01 | Apply the agreed manuscript changes | Editing; wait for explicit author go-ahead | Accepted plan, selected evidence tasks complete |
| W02 | Submission and reproducibility closeout | Packaging; wait for manuscript revision | W01, current TNNLS rules verified |

P01-P03 may proceed independently where their dependencies permit. P04/P05
follow the common indexing contract. E01-E06 are **not** a required omnibus run;
choose only the tests needed for the claims the author retains. A task that
cannot meet its acceptance criteria must narrow the associated claim or report
the remaining blocker, not mark the paper ready anyway.

## P01 - Closest-method comparison

**Purpose:** determine what C1-C3 add, not merely collect citations.

**Inputs:** [literature map](../evidence/LITERATURE.md); active Section III and
Appendix A; [063 METHOD](../../../analyses/063-2026-11-07-architecture-instability-tracing/METHOD.md);
[065 cap geometry](../../../analyses/065-2026-11-07-mlp-frozen-lr-double/observations/010-local-displacement-cap-geometry.md).

**Work/output:** extend the literature map with a formula-level Kosson RRC/cap
comparison and a capability table for Zhu/SVCCA/CKA versus tracing. Search
alternate terms and newer follow-ups. Record exact source versions/locations,
assumptions, the strongest alternative explanation, and supported versus
unverified novelty. Propose a compatible baseline or explain why no faithful
comparison can be made in the selected setting.

**Acceptance:** every proposed novelty statement names its nearest predecessor
and a concrete distinction. Credit BB secants. No unsupported claim of being
first, no field-wide absence inferred from one paper's limitations. No manuscript
bibliography edit in this card.

## P02 - Mathematical and timing contract

**Purpose:** make E03-E07 and E11-E16 use one unambiguous clock.

**Inputs:** [claim audit](../evidence/CLAIMS.md), actual TeX include graph,
research definitions/metrics, Analysis024 observation007, Analysis056/057,
063 method and065 observation013.

**Work/output:** complete a formula ledger: exact, local approximation, stochastic
term, or empirical statistic; state assumptions and denominator behavior. Map
each run field to state, incoming/outgoing update, raw/clipped gradient,
moment/direction/actual increment, and first availability. Audit variable LR,
zero LR, decay, clipping, nonsmooth paths, and the mixed-projection derivation.
Keep the author's notation unless a change resolves a demonstrated ambiguity.

**Acceptance:** no gradient or window fit unavailable at the decision time used
as a predictor; a current gradient can be a feature for a genuinely future target.
no generic Adam `q>2` theorem; exact/all-lag versus lag-one approximation separated.
Check analytic counterexamples and loss-reduction normalizations. Proposal stays
in consolidation records; do not automatically restore previously rejected prose.

## P03 - Exact comparator and evidence inventory

**Purpose:** identify precisely which existing comparisons can answer the paper's
questions and which require rerunning.

**Inputs:** [artifact map](../evidence/ARTIFACTS.md), Analyses024/027/035/037/055/
057/058/059/063/064/065 and their run manifests/configuration hashes.

**Work/output:** a machine-readable CSV alongside its readable summary, with
one row per selected condition: run/attempt, seed, data/model revision, objective,
ordering, fork state, optimizer/moments/clocks, hardware/software, LR, policy
placement, calibration, horizon, probe contract, retained artifacts/hashes, and
complete versus sampled held-out coverage. Identify source files that are only
remote; do not assume all raw evidence survived locally after disk recovery.

**Acceptance:** all proposed tables have unchanged baselines and comparable
denominators. Mark unmatched forks/hardware explicitly. Older TODOs are checked
against completed064/065 work. No remote retrieval or replay is required until
its scope is known; list unavailable artifacts rather than inventing them.

## P04 - Requested high-rate event table

**Purpose:** characterize the chosen three high-LR cases without claiming a
predictor from selected peaks.

**Inputs:** Run215 high-rate records; Analysis058 Figure00/observation000;
Analysis055 direction-context and057 complete-event contracts.

**Proposed contract:** use the existing fixed-pre-rise >10% event definition,
keep censored events, and enumerate all qualifying events within the declared
windows. For a compact main table, choose one example per architecture by an
explicit loss-only rule (for example, first recovered qualifying event), while
retaining the full census. This representative rule is a proposal, not yet frozen.

**Output:** event CSV and table with baseline/onset/peak/recovery states, incoming
versus outgoing update labels, pre-update loss/raw cosine/norm ratio/projected
response, actual `g' increment` where measured, and first availability of each
field. At the step entering the peak, report only information available at its
starting state in the pre-update columns. Endpoint/peak values get separate columns.

**Acceptance:** reproduce known event counts and source metrics; check source
hashes and several state transitions independently. Keep Analysis057's failed
transfer results in supporting context. If the author requests prediction,
route that claim to E06 rather than relabel this table.

**Owner:** reuse the relevant existing event-analysis workspace for numerical
outputs; link them here. Do not allocate a new analysis merely for this plan.

## P05 - Figure and algorithm package from retained evidence

**Purpose:** realize the agreed six-figure storyboard with minimal duplication.

**Inputs:** artifact map and accepted P02/P04 contracts.

**Output:** new publication PDFs and caption/observation records in their owning
analyses; a proposed placement list in consolidation. F4 should combine layer
cosine/RMS, output loss, spectra and selected signed coefficients, borrowing
065's readable stage layout. F5 separates amplitude and reversal under treatment.
T1-T4 may be combined when that improves readability. Algorithms A1/A2 state
costs and output semantics separately.

**Acceptance:** each plotted scalar traces to a source/reduction; source hashes,
window lengths, masks, null values, normalization and sample coverage verified.
MLP450-500 spectra/correlations come from the same refit. Preserve architecture
branches and distinguish selected stage coverage from every internal operation.
Render every new PDF at publication size; keep originals. No manuscript insertion
until W01 is authorized.

## E01 - Matched intervention comparison and replication

**Purpose:** test whether C3 offers value beyond simple damping and existing
representation-change control. This is the recommended new study if C3 remains
a major method contribution; it is not necessary merely to report the historical
case studies with their limits.

**Owner:** continue Analysis065, retaining064 as first-stage evidence. New
executed configurations require new run IDs; read the registry at allocation.

**Proposed design:** MLP/CIFAR-10, ResNet/CIFAR-10 and Transformer/WikiText-2;
three independent evaluation initialization blocks per architecture and two
prespecified fork positions. Use the existing primary forks448/264/552 as the
first candidates; select a second fork by a fixed rule before inspecting new
outcomes. The exact seed values, second forks and horizons remain design fields
to freeze with the author, not choices delegated to outcome-based selection.
Preserve native full-objective training for this first comparison. Practical
minibatch transfer is E04.

**Matched factors:** untreated initial tensors, complete optimizer/RNG/buffer/
data-order state, dataset/model revisions, objective/loss reduction, data order,
precision, hardware/software, diagnostic probe, exposure and evaluation cadence
within each seed/fork block. Every arm includes a same-block unchanged Adam
reference. Match updates/examples/tokens first; wall-time efficiency is separate.

**Candidate arms:** unchanged Adam; fixed local q50 cap; persistent half LR;
raw-gradient clipping before moments; literature-faithful beta2 policy; closest
compatible representation-change baseline from P01. A uniform post-Adam damping
control can distinguish changed optimizer input from changed applied increment.
Norm-matched reference-arm schedules are mechanistic controls, not deployable
competitors. Record the complete intervention, including bias correction and
whether optimizer moments or clocks change.

**Calibration/selection:** fix q50 from discovery evidence before evaluation;
derive thresholds only from untreated pre-fork data. Predeclare equally scoped
tuning budgets for clipping/damping/beta choices. Use training/calibration or
a declared validation split for selection. Existing test results have already
been inspected, so reusing them cannot create an untouched test set by relabeling.
If no new held-out evaluation is available, report this limit and avoid a fresh
generalization-selection claim.

**Capture/outcomes:** every-update training objective, raw/clipped gradient,
moment/direction/applied-increment norms and cosines as needed, local cap scores
and fractions, proposal/actual norms, probe loss/motion. Repeated complete
held-out evaluations at a predeclared cadence, with token weighting for language.
Primary outcome: mean training loss over the entire fixed post-fork window;
also report late/terminal loss, maximum, positive variation, reversal and motion
amplitude, held-out mean/late/terminal results. Report paired per-seed contrasts;
seeds are replication units, not individual updates. Define treatment of the
common untreated fork point before computing maxima/means.

**Retention:** pre-fork calibration and full recovery checkpoints including
optimizer/scheduler/RNG/data position; enough dense representation history for
the declared event windows, rolling histories otherwise. Save source/config
hashes, progress events, manifest and reduction inputs. No unannounced truncation.

**Before execution:** freeze the comparison and outstanding fields; measure a
mechanical smoke/calibration; state live resource price, qualified ETC, maximum
total cost/duration, storage, monitoring and cleanup. Obtain only genuinely
missing scientific/spending authority; do not impose duplicate approvals when
the author has already supplied it. No budget is authorized by this card.

**Acceptance:** exact/qualified baseline replay and observer invariance; all
planned conditions including failures; independent aggregation and figure checks;
download size/SHA-256 verification before worker deletion. A null or negative
result is a valid outcome and must constrain C3.

## E02 - Deployment overhead and efficiency

**Compare:** native training, native plus deployment cap, and dense tracing/audit
as a separate cost. Match device, batches, precision and implementation; use
warmup, synchronized repeated timings, and record peak GPU/host memory and storage.
Count extra forwards, copies and retained histories. Whole-worker runtime is
not isolated diagnostic overhead.

**Output/acceptance:** benchmark config/events plus uncertainty across repetitions,
absolute and relative costs. For speed claims, also compare time to a predeclared
loss target on matched trajectories, with non-reaching runs censored. Do not
infer speedup from lower mean loss at equal update count or larger nominal LR.

## E03 - Probe robustness and observer controls

**Compare:** current ResNet BN group/Transformer sequence prefix against an
independent fixed probe chosen without looking at traces. Preserve full original
normalization groups, sequence masks/context and model mode. Repeat at identical
weights to measure observer noise; preserve training RNG/buffers.

**Output/acceptance:** per-probe loss/motion/spectrum agreement and disagreement,
coverage, IDs, cost and unchanged training checks. An independently selected probe
does not automatically establish whole-population representativeness. Capture
only after retention/storage and replay feasibility are checked.

## Conditional studies: select by the claim

| Card | Comparison and required output | Acceptance / reason to omit |
|---|---|---|
| E04 Ordinary minibatches | Native stochastic training unchanged; separate fixed-probe diagnostics; unchanged Adam versus selected cap and closest controls; matched batches/accumulation/order, independent seed blocks and complete weighted evaluation. | Necessary for a practical minibatch-transfer claim. Must separate sampled native gradients from fixed-objective/probe response. Otherwise bound the paper to full-objective experiments. |
| E05 Spectral specificity and rotation | Matched-length quiet and spike windows selected by a loss-only rule; fresh SVD fits, matched probe/scale, rank/energy summaries. For stationary-axis claims, sliding-window principal angles with sign/degeneracy handling. | Whole-window concentration alone cannot establish spike-specific low dimension or stationary local modes. Omit those stronger claims if not selected. |
| E06 Prospective forecasting | Past-only features/bases/calibration; separate discovery/evaluation trajectories; independently defined future loss event and lead time; all eligible alarms/non-events; loss-history and norm baselines; precision/recall, false alarms, censoring and seed-level uncertainty. | A peak-selected table, retrospectively normalized heatmap, or current-gradient availability is insufficient. This is a separate substantial study; recommended outside the core submission. |

## W01/W02 - Editing and submission gates

**W01:** once the author accepts the plan and explicitly requests application,
edit `manuscript/draft/` only. Update title/abstract/introduction/contributions/
conclusion together; add source-linked methods, approved figures/tables, and
limitations. Preserve supplied originals and abandoned drafts. Verify equations,
citations, table values, source identities, PDF build and all affected pages.

**W02:** verify current official TNNLS requirements for the selected article type;
check length, supplement, declarations, references, figure legibility and final
submission metadata. Prepare reproducibility instructions with code/config/data
identities, complete evidence and limitations. Public release, pushing, and actual
submission require the author's instruction. Local automatic commits still apply
to completed, verified work. The author decides when the consolidation plan and
manuscript are ready; a checklist alone is not an acceptance prediction.
