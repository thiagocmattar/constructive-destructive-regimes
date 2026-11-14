# Intervention ablation TODO

- **Current matrix: complete.** H051 / Runs 293–295 supply all 36 initialization-0 conditions and paper assets in the [selected figure folder](figures/README.md). Exact setup, original estimates and execution receipts remain in the [experimental record](EXPERIMENTAL_PLAN.md).
- **Baseline rule:** every intervention uses unchanged Adam from the same full fork, initialization, objective, runtime and horizon. Settings have unequal effective strength; all outcomes are visible.

| Priority | Work | Exact scope | Status |
|---|---|---|---|
| Complete | Common starts/horizons | MLP 512–800; ResNet 256–576; Transformer 544–928; initialization 0 | All 36 continuations / 11,904 updates verified |
| Complete | Immediate LR | 0.75x/0.5x/0.1x original LR, every model | All 9 treatments verified; no ramp/cosine schedule |
| Complete | Late beta2 | .95/.9/.5, moments/clock retained; native correction | All 9 treatments verified |
| Complete | Global raw-gradient clipping | L2 thresholds .5/1, every model | All 6 treatments verified, including ResNet .5 |
| Complete | Distributed cap | q25/q50/q75 from the same eight untreated transitions before each fork | All 9 treatments verified; q50 remains the preselected main method |
| Complete | Shared measurements and publication outputs | Full loss/stage/update/activity histories; five full held-out states; calibration, audits and cost | Seven PDFs / 20 pages; 36-row outcomes; editable tables/captions and complete reductions |
| Deferred | Initializations 1 and 2 | **72 additional continuations**, 24/model including fresh controls; total study 108 | Faster fleet **6–9 h / ~$49–73**; balanced **10–14 h / ~$47–67**; Spark **~8–12 days** |
| Deferred | Cap quantile x LR x initialization | **45 capped + 9 same-LR controls/model = 54/model; 162 total** | Faster **11–16 h / ~$102–150**; balanced **22–30 h / ~$100–145**; Spark **~21–30 days** |

- **Current execution cost:** provisional **$26.07**, below the authorized $50 ceiling. All task Pods deleted after verified retrieval; **33.842 GB** raw evidence retained on owned Spark; no retained cloud storage. [Receipt](data/matched-compute_closeout.json).
- **Deferred grid assumption:** initializations 0/1/2; LR 1x/2x/4x; q25/q50/q75/q90/q99. Freeze each seed's original-LR cap calibration and compare with its same-LR uncapped control. The grid has not been run here.
- **Qualified reuse:** the current matrix could fill 4 cells/model of the gross future grid (baseline+three 1x caps), leaving 150. After the full three-initialization matrix, 126 would remain. Fork/calibration/measurement equivalence is required; older MLP448–1152 runs are supporting evidence, not substitutes.
- **Cost/ETC limits:** these deferred estimates retain the 14 November planning quote and allowance for prefixes, audits, retrieval and storage. Actual RTX5090 host throughput varies; Spark Transformer extrapolation came from only three mechanics updates. Refresh prices/capacity and measure sustained Spark throughput before any new launch. No extra seeds or grid are required to report the completed single-initialization comparison.
- **Other deferred sensitivity:** intervention time, probe/calibration size, beta2 reset/correction convention, ramped LR and intervention combinations. Broader generalization, stochastic deployment or method-superiority claims need their own tests.
