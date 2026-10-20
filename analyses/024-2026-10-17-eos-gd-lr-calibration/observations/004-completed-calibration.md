# Completed 3k GD learning-rate calibration

All nine conditions reached3,000 full-GD updates. The five new runs122-126
executed sequentially on owned Spark; four conditions reuse verified prefixes
or earlier complete runs. The queue ended October20 at13:47 Sao Paulo time
(16:47UTC), after76.56h. Training and retrieval ETC are zero.

## Sources and verification

`03_results.py --collect` retrieved the final Run126 artifacts and verified
all nine conditions at14:47 Sao Paulo time. Source/config/initial hashes,
loss and gradient coverage, exact q identities, cumulative A/energy/lag-one/
B-squared/B-over-A identities and spectral residuals pass. All254 source
inventory files,26,991 defined adjacent pairs and279 sampled spectral states
are verified. Each condition covers loss states0..3000, gradient states0..2999
and31 spectral states0,100,...,3000. See `results.json`,
`retrieval_verification.json`, `completion_verification.json` and the
[complete nine-condition table](001-available-calibrations.md).

The final source is Run126, `artifacts/attempt-01-spark/production/`:
`metrics.jsonl`, `curvature.jsonl`, `summary.json`, `manifest.json` and its
size/SHA256 transfer inventory. The owning-run retrieval receipt records all32
files (3.017GB). Raw recovery/source evidence remains on both machines.

## Transformer results

| LR | Final training loss | First negative gradient update | Last1k mean cosine | Last1k mean q | Final B/A |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 6.6300970 | Not observed | 0.9999952 | 0.0001963 | 0.7223489 |
| 0.1 | 6.3519354 | 456 | -0.9748604 | 1.9753885 | 0.1395004 |
| 0.2 | 6.0697472 | 10 | -0.9936598 | 1.9936710 | 0.0541329 |

Final Run126 test loss is5.858303 over all245,550 prepared target tokens.
Its final raw norm is0.5826665; final sampled eta*lambda_max is3.120248.
The first negative pair is a sign event, not a persistent-onset definition.
The completed comparison is conditional on one frozen initialization; it
does not establish robustness across initializations or global stability.
q and adjacent cosine are algebraically related. Larger-model spectra use
the declared fixed subsets, not the full training objective.

## Execution, retention and figures

The local30-minute monitor reported SSH exit255 before final retrieval.
A direct read succeeded; its task-specific PID45788 was command-verified
and stopped to avoid concurrent retrieval. Manual collection then passed.
The remote finish-only record confirms no predecessor continuation. At the
14:46 closeout read the GPU had no compute processes,0% utilization,32C
and4.49W. No subsequent training or cloud rental was launched.

The five new runs retain160 inventoried files /6,394,817,635 bytes (6.395GB)
on each machine, below the15GB allowance. This total excludes reused runs,
shared datasets and source packets. Only verified redundant transfer archives
were removed; checkpoints, raw metrics and executed sources remain.

At the status closeout, the existing twelve-panel PDF retained its October19
cutoff of eight completed conditions. The subsequent October20 figure update
adds Run126 and now covers all nine conditions; observation003 records the
final figure and verification. All32 previously plotted curves are unchanged.
No training input or manuscript was changed.
