# Adam author-input revision, 5 November 2026

The supplied proposal is incorporated into the active Adam section with verified
retained-data additions. The current paper develops projected gradient response,
optimizer memory, excursion amplification and recovery. The proposed curvature
comparison is stated as an analytical relationship awaiting neural calibration.

The original [input.md](input.md) and [proposed TeX](05-adam-diagnostics.tex) are
preserved byte for byte. The active source is
[sections/06-adam-diagnostics.tex](../../sections/06-adam-diagnostics.tex), despite
the supplied proposal's `05-` name. [baseline_manifest.json](baseline_manifest.json)
pins all pre-edit source files and 16 preexisting publication figures.

- [Current revised PDF](revised-manuscript.pdf): the same bytes as bare_jrnl.pdf.
- [TODO](TODO.md): implemented items and remaining empirical requirements.
- [Experiment plan](EXPERIMENT_PLAN.md): comparisons, seeds, measurements, retention,
  Spark/local suitability, qualified runtime estimates and live Runpod prices.
- [Analysis057](../../../../analyses/057-2026-11-05-adam-response-memory/README.md):
  new reductions, source hashes, independent verification and publication figures.

## Disposition of the proposal

| Input | Result in the active paper |
| --- | --- |
| Reframe around response, memory, amplification and recovery | Adopted title and sequence; events defined before illustrative examples; intervention-led ending replaced |
| Position relative to current Bai et al. | Published citation and one-HVP baseline checked; no claim that momentum/stochastic experiments are new |
| Give eta*kappa an explicit Adam meaning | z defined as projected gradient response; old figure/table notation explicitly equivalent |
| Add exact mixed-curvature and memory identities | Integrated with assumptions, bound and finite-time alternating transient; synthetic checks pass; no empirical calibration claimed |
| Compare with ordinary oscillation | Added verified Analysis055 low-rate controls with its eight-update definition and scope; duration-matched control stays pending |
| Test incremental alignment information | Eight specifications, seven complete-rate holdouts and coefficient signs recomputed; broad severity claim narrowed |
| Replicate across seeds | Planned three-seed cohort; existing dependent events never counted as independent seed replication |
| Inspect current-gradient cross-alignments | Already retained; new complete-event reduction finds 794/794 locally descending rise-phase updates |
| Report incidence/censoring/threshold sensitivity | Added run-level table and censoring endpoints; all loss-excess and duration summaries retained in Analysis057 |
| Restrict fitted lines to data support | New scatter uses the same 127 events and fits, with observed-range lines and sample counts; original PDF remains unchanged |
| Test HVP/state/stochastic/scale claims | Concrete phased design and costs; no new training or paid execution during revision |
| Consider TNNLS/TPAMI positioning | Stronger explanatory argument adopted; no acceptance prediction or venue-specific experiment requirement invented |

Abstract, introduction, related work, conclusion and the Adam measurement appendix
are aligned with the new scope. The preceding GD/SGD section sources, plots and
loss appendix remain byte-identical; all 16 prior publication PDFs are preserved.
The severity scatter's compiled replacement and a new ablation plot are separate
Analysis057 assets. The new incidence table includes all nine trajectories.

## Evidence and verification

Analysis057 covers 812 candidates (806 recovered, six censored), 131 pronounced
candidates (129 recovered), 127 defined severity events, all 9,216 outgoing
updates and 9,225 loss states. Independent checks cover raw event segmentation,
27 incidence rows, 112 QR regression fits, 2,170 rise/recovery updates, 9,216
cross-alignment states and 2,096 mathematical cases. Full observed support,
negative transfer results and undefined initial diagnostics remain visible.

The ordinary low-rate control counts come from Analysis055's authenticated derived
state table, with its eligibility/horizon explicitly retained. The published Bai
paper was checked for its preconditioner definition, bias correction, momentum,
one-HVP baseline and 187M non-embedding scale. An external-source note is in the
experiment plan; this does not turn that paper's empirical claims into our data.

The PDF compiles without unresolved references/citations or overfull boxes. Final
page/figure visual inspection and hashes are recorded in review.json and
verification.json. Sources and publication PDFs are included in the local commit;
raw caches/checkpoints, scratch renders and unrelated user work are excluded.
The supplied TeX ends with extra blank lines; its original bytes are intentionally
preserved. Whitespace checks exclude that supplied file and pass for authored work.

## Reproduction

From repository root:

```powershell
python analyses/057-2026-11-05-adam-response-memory/01_analyze.py
python analyses/057-2026-11-05-adam-response-memory/02_verify.py
python analyses/057-2026-11-05-adam-response-memory/03_plot.py
python manuscript/draft/revisions/2026-11-05-human-input-adam/02_check_evidence.py
python manuscript/draft/revisions/2026-11-05-human-input-adam/03_estimate_compute.py
python manuscript/draft/revisions/2026-11-05-human-input-adam/01_build_and_verify.py
```

The build uses ignored tmp/pdfs/human-input-adam-2026-11-05 and renders every page
with Poppler. Rebuilding cannot substitute for visual review after layout edits.
The price/availability receipts are dated read-only observations, not live launch
permission or a guarantee of future stock. This task incurred zero cloud cost
and created no cloud resources. Local machine/Spark processes were not changed.
