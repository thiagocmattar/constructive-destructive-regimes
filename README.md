# Constructive and Destructive Training Regimes

Research workspace for revising the constructive-destructive paper and designing
new language-model experiments with Pythia and MiniPile.

The workflow is adapted from
[sparsity-spillover](https://github.com/thiagocmattar/sparsity-spillover).
The workflow began with independent empty registries; no experiment history was
imported. Runs 001–004 now contain verified full-pass Pythia-14M/31M/70M results,
including paired raw/adaptive geometry and a cross-size comparison.
See [research/INDEX.md](research/INDEX.md) for their analyses. The original paper
is context; its existing figures are not results produced in this workspace.

## Start here

1. Read [AGENTS.md](AGENTS.md) and [research/INDEX.md](research/INDEX.md).
2. For paper work, use [manuscript/README.md](manuscript/README.md).
3. Before experiments, settle the definitions in
   [research/DEFINITIONS.md](research/DEFINITIONS.md) and the measurement contract
   in [research/METRICS.md](research/METRICS.md).
4. Follow [research/WORKFLOW.md](research/WORKFLOW.md) for experiment design,
   implementation, execution, retrieval, analysis, and closeout.

```text
AGENTS.md                         collaboration and research conventions
manuscript/draft/                 editable LaTeX extracted from the original ZIP
manuscript/2026-10-03 review/      supplied submission and reviewer materials
research/                        shared definitions, protocols, and current state
research/HYPOTHESES.md            user-directed hypothesis registry
research/findings/               links to verified observations and agreed findings
runs/NNN-YYYY-MM-DD-slug/         one experiment and its attempts/observations
analyses/NNN-YYYY-MM-DD-slug/     independent or cross-run analysis and figures
src/                             shared scientific code, added when justified
tools/                           duration estimates, monitoring, artifact hashes
tests/                           operational checks; scientific tests follow code
data/ and cache/                  ignored local data and derived caches
```

`000-template` folders are templates, never experiments. Runs and analyses have
independent numbering, both beginning at `001`. Completed runs preserve their
executed inputs, artifacts, observations, and cloud closeout records.

## Local verification

Operational helpers use the Python standard library; scientific checks also
require the run's PyTorch dependencies. From the repo root:

```text
python -m unittest discover -s tests -v
```

The connected Runpod MCP and native SSH/SCP provide the cloud control and transfer
route. An ignored local `runpodctl` binary supports the deadline guards. See
[research/RUNPOD.md](research/RUNPOD.md). Each run pins and records its training
environment. Migration scope and provenance are in
[research/BOOTSTRAP.md](research/BOOTSTRAP.md).
