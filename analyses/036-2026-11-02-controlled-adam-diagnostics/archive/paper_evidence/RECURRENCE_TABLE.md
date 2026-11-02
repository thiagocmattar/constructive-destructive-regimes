# Descriptive recurrence across all nine arms

| Case | LR | Episodes | Earlier crossing: all | Large episodes | Earlier crossing: large | Lead for large: median [range] |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLP | 0.0001 | 8 | 5/8 | 0 | -- | -- |
| MLP | 0.0003 | 46 | 18/46 | 13 | 11/13 | 7 [3,8] |
| MLP | 0.001 | 39 | 17/39 | 17 | 13/17 | 5 [1,8] |
| ResNet-32 | 0.0003 | 0 | -- | 0 | -- | -- |
| ResNet-32 | 0.001 | 10 | 9/10 | 4 | 4/4 | 1.5 [1,6] |
| ResNet-32 | 0.003 | 7 | 5/7 | 3 | 3/3 | 2 [1,3] |
| Transformer | 0.0003 | 0 | -- | 0 | -- | -- |
| Transformer | 0.001 | 6 | 5/6 | 1 | 1/1 | 5 [5,5] |
| Transformer | 0.003 | 13 | 13/13 | 7 | 7/7 | 5 [2,8] |

Loss-only episodes begin at a 10% rise above the preceding 16-state minimum. Large means an observed peak at least 50% above that baseline. Earlier crossing means a fresh crossing of raw projected change above two, 1-8 updates before onset while loss is less than 5% above its own recent minimum. Every episode remains in the denominator, including misses. Leads condition on a hit and use the earliest qualifying crossing in that window. The highest-rate MLP and Transformer each include one unfinished recovery. These are dependent events from one initialization per case, not independent replications.

The zero-episode low-rate ResNet and Transformer arms nevertheless cross two during training. Thus crossing two is not sufficient for a pronounced excursion. This table describes recurrence; it is not a classifier evaluation.
