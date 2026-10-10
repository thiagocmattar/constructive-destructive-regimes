# Workflow adaptation and provenance

Setup date: 2026-10-03. Reference:
[thiagocmattar/sparsity-spillover](https://github.com/thiagocmattar/sparsity-spillover).

Inspected the adjacent local checkout, whose `origin` matches that URL, at commit
`93dba1b325b78f614c7bb8fa37465985fa1d3f15` (2026-09-25). The inspected tracked
workflow files were unchanged against that commit. The checkout had unrelated
untracked prelaunch scripts; none were copied. GitHub API reads confirmed the
repository and returned a README blob matching the local file
(`3e8fcc12efc69e9d7d27161fb2d08615b32bcf7c`). This is a pinned workflow adaptation,
not a claim that every remote file was audited.

## Carried forward

- Numbered experiment folders and independently numbered analysis folders.
- Run-local configs/scripts/evidence, separate infrastructure attempts, observation
  indexes, PDF figures, and consolidated findings.
- Compact shared definitions/methods/data/metrics/manuscript/compute documentation.
- Production-shaped calibration, transparent ETC arithmetic, detached jobs,
  read-only monitoring, hash-verified retrieval, and resource/billing closeout.
- Small operational helpers. ETC and monitoring helpers are adapted from the
  source's `tools/estimate_etc.py` and `tools/watch_run.py`; artifact inventory
  implements the source workflow's transfer-verification practice.

## Adapted deliberately

Hypothesis/run/analysis/finding registries are empty. Scientific methods and
metrics are undecided. No sparsity pressure/gates/kernels, training engine,
experiment history, source paper, credentials, caches, or weights were copied.
No scientific defaults for AdamW, clipping, batch size, seed, horizon, or model
initialization were inherited.

The source's fixed two-confirmation workflow was adapted to honor authorization
already present in a session and to ask only about material scientific ambiguity
or missing execution/spending authority. Reversible preparation is completed
before any necessary launch question. Existing authority is not reset by a folder
template or document.

Training dependencies are deferred until a concrete metric/workload design.
Bootstrap tools are standard-library Python and can be tested without installing
the source study's GPU environment.

## Setup evidence and limits

- Python 3.12.10 and Git 2.53.0.windows.3 are available.
- Runpod MCP authenticated on a read-only Pod inventory; no cloud mutation made.
- Native SSH/SCP and WSL are present. `runpodctl` was not on Windows PATH.
- LaTeX tools are present, but no compile or paper-content edit was performed.
- The working draft's 22 extracted files initially matched archive bytes; their
  baseline identities are in `manuscript/source_inventory.json`.

Local operational verification is recorded in `tests/`. This validates the
workflow helpers, not any future training algorithm or scientific measurement.
The setup receipt is in `bootstrap-verification.json`. Git was initialized
locally; there is no remote configured or initial commit.
