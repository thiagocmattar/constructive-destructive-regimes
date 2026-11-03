# Analysis038: Adam spike intervention design

Status: proposed for author evaluation. H028; no intervention training or paid
execution. The requested comparison is whether interventions reduce the three
selected Adam loss excursions while preserving optimization progress.

The recommended first experiment has eight arms at each of three matched forks,
32 updates before the formal pre-rise state. Each arm continues for 128 updates.
It compares two LR reductions, raw-gradient clipping, Adam-step clipping,
removing the momentum numerator, faster second-moment adaptation, and a
full-objective acceptance safeguard against unchanged Adam.

- [Protocol and method definitions](PROTOCOL.md)
- [Source audit and implications](observations/001-source-audit.md)
- [Selected events and proposed forks](cases.csv)
- [Extended scalar context](extended_context.csv), covering 96 updates before
  each formal pre-rise state through 128 updates after it
- [Source identities and verified reductions](source_review.json)
- [Design verification](verification.json), including independent scalar checks
  of descent/overshoot, Adam clipping, history-aware bias correction and Armijo

The raw histories show negative gradient dot applied-step products throughout
all 28 pre-rise-to-peak updates. A safeguard triggered only by an uphill
first-order direction would not act on these untreated rises. Finite-step
overshoot remains a candidate explanation, not an identified curvature result.

Run215 retained final recovery states but overwrote its intermediate recovery
file. New matched prefixes must therefore be saved before testing interventions;
the historical peak cannot be assumed to replay on another device.

Reproduce the read-only source reduction from the repository root:

```powershell
python analyses/038-2026-11-03-adam-spike-intervention-design/01_audit_sources.py
```

The script reads existing evidence and writes these analysis-owned CSV/JSON
reductions. It does not evaluate a model or change any source history. Method
implementations, calibration, new run IDs and a spending envelope follow design
review. Historical cloud spending authority is not carried into this proposal.
