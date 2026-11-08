# Severity 10pct

Signal at state s, individual relative loss increase >=10% at s→s+1. Counts are eligible updates, not independent spike events. High-LR cases only; the complete data/severity.csv retains non-target alarms, precision and all thresholds.

| Case | Rises | rho<0 | r>1 | Both | Q>=2 |
|---|---|---|---|---|---|
| mlp | 168 | 149 | 110 | 105 | 82 |
| resnet | 12 | 12 | 11 | 11 | 7 |
| transformer | 26 | 26 | 22 | 22 | 22 |
