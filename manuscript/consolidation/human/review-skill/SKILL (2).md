---
name: research-framing
description: >-
  Position academic or technical research within prior literature: map related work,
  verify consequential research gaps, articulate defensible novelty and contributions,
  and draft introductions, related-work sections, limitations, and future-work agendas.
  Use for framing an observation, experiment, definition, method, theorem, benchmark,
  replication, or project proposal; for revising a paper's positioning; and for
  auditing gap-to-evidence alignment. Do not trigger for ordinary proofreading,
  isolated paper summarization, or citation formatting without a positioning task.
---

# Research Framing

Turn a research result or idea into a **literature-grounded, evidence-calibrated argument**. Keep distinct what is **established**, **inferred**, and **untested**. Persuasion must follow—not replace—literature verification and scrutiny.

## When to use

Use when positioning experiments or definitions, verifying research gaps, comparing explanations, identifying contributions, drafting an introduction or related work, planning specific future research, or auditing whether a narrative oversells the evidence. Begin even from a preliminary idea or experiment. If the user requests only one deliverable, do not force the entire framework.

## Resources

Read supporting files selectively:
- `references/literature-research.md` for finding/screening papers, extracting evidence, and validating gaps.
- `references/argument-patterns.md` for reusable argument moves, framing alternatives, and writing formulas.
- `references/claim-audit.md` for falsification tests and distinctions among empirical, causal, predictive, and theoretical claims.
- `assets/framing-worksheet.md` for the reusable exercise template.
- `assets/literature-matrix.csv` for structured paper comparisons.

## Operating rules

1. Ground the analysis in actual project material. Preserve terminology and uncertainty rather than repairing gaps silently.
2. Find nearest neighbors, including work under alternate names and later follow-ups. Give prior work full credit. An author's future-work paragraph is a search lead, not proof of a literature-wide gap.
3. Prefer **specific, consequential, verified** gaps to generic claims that a topic is “underexplored.”
4. Do not describe algebraic restatements as independent validation; correlation as causation; heuristics as theorems; retrospective detection as forecasting; or exploratory results as confirmation.
5. Frame the strongest **supported** contribution, even if it is a replication, extension, measurement, negative result, or limitation of previous work.
6. Explicitly state assumptions, comparators, conditions, alternative interpretations, and the boundary of each claim.
7. Cite primary sources and exact locations for material literature claims. If research access is unavailable, label novelty and gap assessments provisional; never invent citations.

## Research workflow

The **discovery order** differs from the **presentation order**: first identify what the work shows, then verify prior art and gaps, then write stakes → prior knowledge → gap → approach → evidence → contributions and scope.

### 1. Inventory the research object

Record: question; work type (definition/experiment/theorem/method/benchmark/replication); result; setup and assumptions; comparators; measured evidence; alternative explanations; and status (proposed, exploratory, validated, proved).

Use this factual sentence:

> Under **[conditions]**, we **[observed / established / enabled] [result]**, relative to **[comparator]**, using **[evidence]**. This supports **[bounded conclusion]**.

If information is missing, proceed with `unknown` labels rather than forcing a questionnaire.

### 2. Map literature around questions, not labels

Identify: closest direct papers; theoretical foundations; competing explanations; baselines; contradictory results; reviews; and newer follow-ups. Search alternative terminology; check references backward and citations forward when practical. For every key paper, record its exact question, result, assumptions, boundary, version/date, source location, relation to this project, and evidence status. Read `references/literature-research.md` for the method.

### 3. Generate and validate candidate gaps

For every candidate, specify **established knowledge**, **unresolved question**, **documented evidence of the boundary**, and **why solving it matters**. Verify whether newer work addresses it. Distinguish paper-specific limitations from field-level gaps.

> Prior work **[A, B]** establishes **[X]** under **[conditions]**. However, **[specific Q]** remains unresolved in **[target setting]**, as indicated by **[source/boundary]**. Resolving it matters because **[consequence]**.

If no gap survives, say so and recommend a narrower question or additional literature research.

### 4. Compare 2–3 plausible framings

Candidate relations: extension; mechanism; reconciliation; definition/measurement; method/capability; negative/boundary result; replication; or resource/benchmark. Compare importance, difference from closest work, available evidence, and defensible scope. Select the strongest evidence-backed narrative, not the grandest one.

### 5. Align gap → contribution → evidence

Create a table:

| Gap/question | Closest prior work | Actual contribution | Evidence/test | Conditions and caveats |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

Every major contribution must answer an explicit question and have a supporting result. Use exact verbs: **propose**, **observe**, **support**, **show experimentally**, **prove**, **predict**. Provide strongest alternative interpretation and decisive control/experiment/counterexample.

### 6. Draft the narrative

Present: **problem and stakes → strongest prior knowledge → specific unresolved question → conceptual/methodological change → evidence and scope → distinct contributions → limitations and future work**. Related work should be grouped by *role in the argument*, not a chronological citation dump. Credit the single closest predecessor explicitly. Draft only the depth requested. Consult `references/argument-patterns.md`.

### 7. Adversarial audit

Apply `references/claim-audit.md`. Ask whether novelty survives the closest and latest papers; whether thresholds were selected independently; whether causal claims distinguish alternatives; whether proof assumptions match actual algorithms; whether online prediction uses only available information; whether an allowed counterexample defeats a theorem; and whether limitations genuinely constrain claims. Revise the claim, not just its prose, when a test fails.

## Default output for a full framing task

1. **Research object/status:** factual summary; established/inferred/untested.
2. **Literature map:** key verified references, results, scopes, and boundaries.
3. **Candidate gaps:** 2–3 options, their support, and selected gap.
4. **Positioning thesis:** a sentence and polished paragraph.
5. **Contribution–evidence matrix:** bounded claims, comparators, interpretations.
6. **Introduction/related-work outline or draft:** as requested.
7. **Limitations and next tests:** uncertainties tied to concrete experiments/derivations.
8. **Evidence note:** search date, verified source locations, and provisional judgments.

The final argument should make these answers obvious: **Why this question? What is known? What is unresolved? Why does it matter? What changes here? What evidence supports it? Where does the claim stop?**
