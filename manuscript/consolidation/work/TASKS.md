# Closure tasks for the current paper plan

- **Authority:** [PLAN.md](../PLAN.md). This file is an implementation index, not a second plan.
- **Current phase:** evidence consolidation is complete; manuscript edits and new experimental execution await the author's next instruction.
- **Completed:** repository/literature review; exact clock and claim audit; Adam sign/severity/forward-memory analysis; retained-Gram controls; section-indexed asset package and compact tables.

| Plan step | Work | Inputs | Completion condition |
|---|---|---|---|
| 1. Assemble existing evidence | Revise I–III in place, retain IV Figures 1–4, integrate compact V-A and V-B/C methods/results. | Section folders under evidence/, algorithms/equations, four contribution cards. | Every retained claim has the correct source, clock, coverage and caption; final Figures 5–7 assembled without changing scientific reductions. |
| 2. Qualify comparisons | Audit starting tensors, optimizer/RNG, objective, runtime, probe and calibration; reuse qualified branches or prepare paired repeats. | V-C records and baseline/cap table. | Comparable results identified explicitly; unmatched historical figures are not presented as a ranking. |
| 2, if execution is needed | Native Adam, native half-LR, calibrated raw-gradient clipping, distributed q50 and same-layer norm-schedule control per model; late beta2=.95 additionally for MLP. | MLP 448–1152/LR .001; ResNet 264–576/.003; Transformer 552–928/.003; eight-update untreated calibration. | Maximum 16 seed-0 continuations or 48 over three initializations before reuse; unchanged baseline retained; absent excursions and all failures reported. |
| 3. Complete reporting | Controller activity/bounds, probe and held-out coverage, matched baseline/controller time and peak memory. | Existing records first; new measurements only where missing. | Complete evaluation at start/start+32/end and clearly separated diagnostic/tracing/controller costs. |
| 4. Conditional extensions | Validated forecasting, quiet-window/independent-probe contrasts, causal source, superiority to representation-aware methods, stochastic/scale transfer. | Exact additional test specified in PLAN. | Required only for the corresponding stronger claim; not a default prerequisite to the explanatory paper. |
| 5. Finalize | Abstract, conclusion, citations, compilation/rendering, reproducibility and submission route/format. | Completed manuscript and verified local package. | Scientific claims and display captions agree; venue route is resolved before any submission. |

- **Execution home:** continue Analysis 065 for interventions; changed scientific inputs get new run IDs. Do not create an independent intervention analysis.
- **Launch preparation:** quote live cost, qualified duration, maximum envelope, storage, monitoring and retrieval checks before billable work.
- **Package maintenance:** run `python manuscript/consolidation/work/package_evidence.py --verify`. Run `--build` only after reviewing the catalogue and intended source changes; it refreshes copies and compact tables.
- **No automatic promotion:** existing source-report numbering, historical prose and exploratory findings are not automatically final manuscript content.
