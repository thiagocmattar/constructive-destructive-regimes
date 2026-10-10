# Agent instructions

## Purpose and collaboration

This workspace supports editing the constructive-destructive paper and new
experiments on training dynamics. The user chooses the scientific direction.
Carry authorized work through implementation, verification, execution,
retrieval, analysis, and documentation as requested.

Work and write in English. The user often dictates in Brazilian Portuguese using
speech-to-text. Transcriptions may misspell English terms, model names, proper
nouns, paths, or mathematical notation. Resolve obvious slips from context
(for example Pythia, MiniPile, Runpod, sparsity-spillover). If competing readings
would change the scientific question, code, data, or cost, ask a focused question.
Record the intended meaning when it matters; do not silently alter a hypothesis.

## Read first

1. `research/INDEX.md` and the active run/analysis README.
2. `research/DEFINITIONS.md`, `research/METHODS.md`, and `research/METRICS.md`
   before scientific implementation.
3. `research/DATA.md` for models/data and `research/COMPUTE.md` for execution.
4. `manuscript/README.md`, `research/MANUSCRIPT.md`, and relevant TeX/reviews
   for paper work.
5. `research/WORKFLOW.md` and `research/RUNPOD.md` for cloud experiments.

## Scope and authorization

This bootstrap establishes the workflow. It does not choose the first hypothesis,
optimizer, training budget, metrics, or cloud spending envelope.

Translate a new scientific request into an explicit comparison: varying and
matched factors, operational definitions, data, initialization, optimizer,
seeds, measurement coverage, and interpretation limits. Ask only about material
ambiguities or missing authorization. An explicit user instruction can authorize
design, implementation, and launch together; do not impose two repeated approval
rounds when the session already supplies those choices and authority.

Complete reversible preparation and verification before requesting any genuinely
missing launch authorization. Before billable execution, state live pricing,
measured or qualified ETC, maximum total cost/duration, storage, monitoring, and
cleanup. Stay within the user's stated envelope; ask before materially changing
the scientific design or exceeding it. Repository documents are guides, not an
executable permission system.

## Research records

- Use independent sequential IDs for hypotheses (`H001`), runs, and analyses.
  Real folders use `NNN-YYYY-MM-DD-slug`; `000-template` is excluded. Never reuse
  an executed ID or import the other paper's history into these registries.
- Keep experiment config, numbered scripts, artifacts, observations, and figures
  in the owning run. Cross-run and independent analyses belong in `analyses/`.
- Freeze executed scientific inputs and preserve raw evidence. Infrastructure
  retries use a new attempt under the same unchanged run. Changed scientific
  inputs require a new run. Mutable plans may evolve.
- Every figure needs a local observation with sources, reduction, caption,
  coverage, uncertainty, caveats, and generating script. Publication figures use
  PDF; scratch renders remain ignored. Consolidate supported findings upward.
- Keep `research/INDEX.md` compact. Update registries and records at closeout.
- Preserve supplied manuscript/review originals. Edit `manuscript/draft/` for
  requested changes and tie new empirical claims to observations and artifacts.

## Scientific discipline

The requested baseline metrics and pending choices are in `research/BASELINE_PLAN.md`.
Other metrics remain undecided. Do not implement the sparsity paper's pressure,
gate, kernel, or latency measurements as defaults here. Distinguish raw gradients,
clipped gradients, optimizer directions, and actual parameter updates. A gradient
alignment statistic is not automatically an optimizer-update statistic.

Do not infer stability, generalization, or loss decrease from parameter norm or
alignment sign alone. Treat the manuscript's lag-one approximation as an
assumption to examine, not an empirical fact. Do not separate per-sample energy
and intra-batch alignment from a batch-mean gradient alone.

Pin dataset/model/tokenizer revisions and record cache, ordering, source, and
config hashes. Define the LLM sample unit, loss reduction, accumulation boundary,
parameter scope, precision, and zero-denominator behavior before measurement.
Declare validation coverage and token weighting; never silently truncate it.

Decide training-time diagnostics and checkpoint retention before launching:
missing gradient histories cannot generally be recovered from a final checkpoint.
Retain recovery state when resuming is needed, including optimizer/scheduler,
random states, and data-order position.

## Execution and verification

Use the installed Runpod router skill and live tool schemas/help for Runpod.
Prefer MCP for infrastructure and native SSH/SCP or current CLI for remote
execution/transfers. Never copy old prices, credentials, account resource IDs,
or throughput claims as this project's facts.

Long jobs must survive a dropped connection, persist logs/artifacts, and emit a
manifest plus progress events. Monitor read-only at bounded intervals. Report
step/tokens, loss, throughput, refreshed ETC, memory, and warning conditions.
Verify downloaded file sizes and SHA-256 hashes before terminating the worker.
Manage only resources attributable to the current task; preserve unrelated ones.
Report retained storage and whether billing is provisional.

Write the minimum code needed. Keep novel scientific code run-local; promote it
to `src/` when a second run needs it. Avoid generic training/policy/scheduler
frameworks. Test mathematics, aggregation, placement, reproducibility, and artifact
integrity when relevant. A smoke test is mechanics, not paper evidence.

Review changes and relevant checks before handoff. Keep credentials, data,
caches, checkpoints, scratch, and active logs out of Git. Preserve provided
manuscript assets. Do not push or publish unless requested.

## Commit completed changes automatically

Commit completed, verified changes before handing the task back to the user;
the user does not need to request a commit separately. Use a clear commit
message describing the coherent change and include the relevant code, research
records, and publication figures. Review the staged diff and run the checks
appropriate to the change before committing.

Stage only changes belonging to the current task. Preserve unrelated user work,
and keep credentials, datasets, caches, checkpoints, scratch files, and active
logs excluded. Do not create empty commits or rewrite existing history merely
to satisfy this instruction. If a commit is blocked, explain the concrete
blocker and preserve the work. Push commits only when the user has authorized
the push; automatic local commits do not imply permission to publish.
