# 01: Persistent immediate half LR

Sources: Analysis044 `results.json`, `figure_values.csv`, `FINDINGS.md`; Run223
prefix/control/fixed_half boundary records. Generator `../01_figures.py` uses
the exact raw records and excludes cosine treatments. Independent read-back
also matches the original `fixed_half` plot values.

Caption: Immediate permanent halving of the learning rate (blue) versus
unchanged full-batch Adam (gray), from the same warm model and optimizer state
within each column. Columns are CIFAR-10/MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer. Rows show complete training-objective loss (log scale),
projected gradient change, raw-gradient cosine, raw-gradient norm and applied
LR divided by base LR. Gray bands show shared pre-intervention context;
vertical lines mark the fork. Curves are unsmoothed, with independent scales.

Coverage: MLP states 448--800, fork 512, base LR 0.001; ResNet states 192--576,
fork 256, base 0.003; Transformer states 480--928, fork 544, base 0.003. Response
windows contain 288/320/384 updates. The objectives use 5,000 MLP images,
50,000 ResNet images and the original token-weighted Transformer corpus.
Controls are fresh same-worker controls; failed historical replay is not reused.

Result: immediate half changes mean training loss by -25.04%, +27.20% and
-9.66%, respectively. MLP loses its detected >10% excursions in this window;
ResNet and Transformer retain them. No cross-model average is computed.
One seed and selected windows provide conditional intervention responses;
neither smoother diagnostics nor lower finite-window loss proves convergence,
generalization, or diagnostic-guided timing value. See the shared observation
contract for exact clocks, reduction and uncertainty.
