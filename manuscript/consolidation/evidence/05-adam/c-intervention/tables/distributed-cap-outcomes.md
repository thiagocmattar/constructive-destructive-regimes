# Distributed cap outcomes

Distributed q50 cap relative to its own unchanged Adam baseline; negative is lower. Mean/tail32/max/PV use the stated training-loss states; test is complete terminal held-out loss. MLP has three seeds; larger models have one. This is not a matched ranking against historical LR/clipping/beta2 studies. Absolute paired values and exact source worker identities are in distributed-cap-pairs.csv.

| Case | Seed | States | Mean % | Tail32 % | Max % | PV % | Test % |
|---|---|---|---|---|---|---|---|
| mlp | 0 | 448--1152 | -22.85 | +4.36 | -39.04 | -85.49 | +2.87 |
| resnet | 0 | 264--576 | -16.22 | -7.48 | -13.82 | -46.78 | +8.18 |
| transformer | 0 | 552--928 | -26.00 | -73.36 | -79.63 | -98.81 | +100.17 |
| mlp | 1 | 448--1152 | -19.43 | -28.52 | -19.96 | -73.92 | +2.76 |
| mlp | 2 | 448--1152 | -32.08 | -64.49 | -51.62 | -82.56 | +6.13 |
