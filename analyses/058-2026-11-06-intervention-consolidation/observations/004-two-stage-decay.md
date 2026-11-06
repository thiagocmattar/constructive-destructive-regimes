# 04: MLP two-stage cosine decay

Sources: Analysis050 `FINDINGS.md`, `results.json`, `plotted_values.csv`;
Runs223/226/227. Generator `../01_figures.py` preserves both source comparisons.

Caption: Two 32-update cosine reductions for CIFAR-10/MLP: base LR 0.001 to
0.75x over states 512--544, then to 0.5x over 650--682 (green), compared with
unchanged LR (left) and the matching first reduction held at 0.75x (right).
Rows show complete training loss (log), projected gradient change, raw cosine,
raw norm and applied/base LR. Gray bands show pre-512 context; dotted lines
mark the second reduction's start and completion. Independent panel scales;
no smoothing or truncation of the selected window.

Coverage: states 448--800; all 288 post-512 updates and 150 post-650 updates;
full 5,000-image objective. The treated and held-0.75x paths match until 650.

Result: relative to held-0.75x, the first post-650 peak is 48.77% smaller and
recovers 18 updates earlier. Mean loss over 651--800 is 27.66% lower. The first
spike remains; no >10% excursion peaks after 682 within the observed window.
Timing was selected after earlier inspection. This is controlled development
evidence for one seed/window, not independent confirmation, a universal safe
floor or a held-out improvement. No new event reduction or uncertainty estimate.
