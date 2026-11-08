# Next step signs

Strict outgoing increases/decreases after the decision-time signal. Each signal cell counts flagged increases / flagged decreases; no loss-magnitude threshold. All nine arms, incoming controls and tolerance checks remain in data/one-step-summary.csv.

| Case | Increases | Decreases | rho<0 up/down | r>1 up/down | Both up/down | Q>=2 up/down |
|---|---|---|---|---|---|---|
| mlp | 454 | 569 | 404/412 | 261/285 | 239/218 | 172/136 |
| resnet | 95 | 928 | 86/214 | 47/277 | 46/116 | 21/15 |
| transformer | 152 | 871 | 145/376 | 99/322 | 98/203 | 81/66 |
