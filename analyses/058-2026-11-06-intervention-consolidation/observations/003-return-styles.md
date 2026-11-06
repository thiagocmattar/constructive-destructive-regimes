# 03: MLP return styles

Sources: Analysis046 `FINDINGS.md`, `results.json`, `plotted_values.csv` and
Runs223/224/225. Generator `../01_figures.py` preserves all three source
comparisons; only styling and row order change.

Caption: CIFAR-10/MLP with a 32-update cosine reduction to half LR starting at
512, a hold until 650, and a 70-update cosine increase to 0.75x ending at 720
(green). Columns compare this trajectory with unchanged base LR, the same
cosine reduction held at half, and abrupt restoration to 0.75x at 650.
Rows show log training loss, projected gradient change, raw cosine, raw norm,
and applied/base LR. Gray shading precedes the initial intervention at 512;
dotted lines mark 650 and 720. All curves are unsmoothed.

Coverage: states 448--800, base LR 0.001, complete 5,000-image objective.
The three restoration/held-half paths match through 650. Unlike Figure 02's
right panel, these comparisons deliberately use the cosine-half history.

Result: the gradual return delays the large excursion but gives a 2.60x higher
peak than abrupt return. Mean loss over 651--800 is 33.75% above abrupt return
and 94.43% above held-half. Fewer recovered-event counts do not imply better
spike suppression when a long event contains multiple peaks. One seed and a
selected return time/duration do not establish a general schedule rule; endpoint
test loss and training-loss behavior are distinct. No new uncertainty interval.
