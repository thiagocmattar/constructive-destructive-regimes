# Research workflow

## 1. Interpret the request

Read the relevant manuscript, current research state, and user request. Correct
obvious dictation slips from context. Resolve material ambiguity about scientific
meaning rather than inventing a new research direction.

Write a concrete comparison: question, matched/varying factors, operational
definitions, model/data, initialization, optimizer, seeds, budgets, diagnostic
coverage, expected support/refutation evidence, and limits. Reuse authorization
already given in the conversation. Ask only for missing scientific choices or
execution authority.

## 2. Allocate and implement

Use the next independent run or analysis number and the local date:
`NNN-YYYY-MM-DD-slug`. Copy the appropriate `000-template` and replace placeholders.
Templates are not run history. Record the request and resolved interpretation.

Keep numbered scripts beside their config and evidence. Add only needed files;
do not create empty training/plotting scripts. Analysis may be independent of runs
or combine several runs; it owns its reductions, figures, and observations.

Implement scientific measurement only after its contract is explicit. Run focused
tests and a labeled smoke check. Measure production-shaped memory, complete update
times, validation, diagnostics, checkpoints, and transfer costs. Small synthetic
tests and abbreviated smokes are not empirical paper evidence.

## 3. Prepare execution

Produce a reviewable launch packet with code/config identity, tests, calibration,
ETC and uncertainty, local/cloud choice, artifact/checkpoint inventory, monitoring
cadence, warning conditions, and recovery plan. For cloud, include live price,
maximum aggregate cost/duration, worker count, storage, deadline, and teardown.

If the request already authorizes the concrete design and spending envelope,
proceed within it. Otherwise ask for the genuinely missing decision after all
authorized preparation is complete. A paid calibration also needs to be within
the spending authority. No resource is provisioned just to complete this bootstrap.

## 4. Execute and monitor

Create an attempt with resolved config and manifest. Run detached with persistent
logs and events. Monitor read-only at bounded intervals, reporting progress,
loss/diagnostics where defined, throughput, memory, refreshed ETC, and warnings.
Do not confuse Pod/process presence with training health or terminal success.

An infrastructure retry stays in the same run if scientific inputs are unchanged;
record the failure and new attempt. Changed scientific inputs receive a new run.
Do not rewrite executed configs or raw evidence.

## 5. Retrieve and close compute

Complete the agreed diagnostics and save the required training/recovery states.
Create a relative-path, byte-size, SHA-256 transfer inventory. Copy artifacts and
verify them locally before removing their worker. Each worker must have its own
verification gate; cohort verification follows once all workers are retrieved.

Terminate task-owned compute, confirm state, and report intentionally retained
storage. Distinguish estimated cost from posted provider billing and identify
incomplete billing windows. Details: [RUNPOD.md](RUNPOD.md).

## 6. Analyze and consolidate within scope

Carry out the analysis/figures/manuscript edits included in the user's request.
Do not launch unrelated follow-up experiments. Each figure/result needs an
observation, sources, reduction, coverage, uncertainty, and caveats. Preserve
negative results and failed qualification.

Update the run/analysis README and index, hypothesis status if applicable, and
compact research index. Promote supported findings with user agreement on the
interpretation. New manuscript claims must trace to the originating observation.

## Version control

Review scoped changes and checks before handoff. Track code, configs, records,
compact evidence, inventories, and publication figures. Ignore credentials,
datasets, caches, weights, active logs, transfer archives, and scratch. Preserve
the original supplied manuscript archive/PDFs. Never push without a request.
