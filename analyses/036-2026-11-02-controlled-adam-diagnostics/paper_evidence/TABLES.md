# Evidence tables


Generated from verified histories. Episode counts and state forecasts have different denominators. Lead summaries are conditional on a hit, in completed updates before the 10% onset. Repeated observations are dependent; these ranges are descriptive, not confidence intervals.


## A. All nine arms: episode recurrence and early warnings


| Case | LR | Episodes | Large | Crossing: all | Crossing: large | Lead: large median [range] | Gradient growth: large | Momentum growth: large | Unrecovered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.0001 | 8 | 0 | 5/8 | -- | -- | -- | -- | 0 |
| MLP | 0.0003 | 46 | 13 | 18/46 | 11/13 | 7 [3,8] | 12/13 | 13/13 | 0 |
| MLP | 0.001 | 39 | 17 | 17/39 | 13/17 | 5 [1,8] | 13/17 | 13/17 | 1 |
| ResNet-32 | 0.0003 | 0 | 0 | -- | -- | -- | -- | -- | 0 |
| ResNet-32 | 0.001 | 10 | 4 | 9/10 | 4/4 | 1.5 [1,6] | 3/4 | 3/4 | 0 |
| ResNet-32 | 0.003 | 7 | 3 | 5/7 | 3/3 | 2 [1,3] | 3/3 | 3/3 | 0 |
| Transformer | 0.0003 | 0 | 0 | -- | -- | -- | -- | -- | 0 |
| Transformer | 0.001 | 6 | 1 | 5/6 | 1/1 | 5 [5,5] | 1/1 | 1/1 | 0 |
| Transformer | 0.003 | 13 | 7 | 13/13 | 7/7 | 5 [2,8] | 7/7 | 7/7 | 1 |


A fresh crossing means projected change moves from <=2 to >2. Norm-growth alarms require >20% one-state growth. Warnings must occur 1-8 states before onset while loss is <5% above its own trailing minimum. A large episode reaches >=50% above its frozen loss baseline. The CSV also retains all-event norm alarms, joint alarms, IQRs and misses.


## B. Specificity: fixed >2 alarm on the forward-test states


| Case | LR | States | Positive | Alarms | True | False | Precision % | Recall % | FPR % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 0.0001 | 562 | 47 | 217 | 38 | 179 | 17.5 | 80.9 | 34.8 |
| MLP | 0.0003 | 311 | 146 | 68 | 43 | 25 | 63.2 | 29.5 | 15.2 |
| MLP | 0.001 | 239 | 126 | 28 | 19 | 9 | 67.9 | 15.1 | 8.0 |
| ResNet-32 | 0.0003 | 633 | 0 | 175 | 0 | 175 | 0.0 | -- | 27.6 |
| ResNet-32 | 0.001 | 505 | 59 | 32 | 11 | 21 | 34.4 | 18.6 | 4.7 |
| ResNet-32 | 0.003 | 613 | 7 | 2 | 2 | 0 | 100.0 | 28.6 | 0.0 |
| Transformer | 0.0003 | 633 | 0 | 190 | 0 | 190 | 0.0 | -- | 30.0 |
| Transformer | 0.001 | 553 | 43 | 94 | 28 | 66 | 29.8 | 65.1 | 12.9 |
| Transformer | 0.003 | 485 | 67 | 85 | 46 | 39 | 54.1 | 68.7 | 9.3 |


An alarm is a state, not an independent episode. False means no >=10% rise relative to current loss within8 updates; it does not mean no later rise at any horizon. Zero-positive arms have undefined recall. The same 4,534 decisions are used for every primary comparison.


## C. Primary incremental prediction


| Case | States | Positive | Prevalence % | AP: loss | AP: + projected | Delta AP | Brier: loss | Brier: + projected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | 1112 | 319 | 28.7 | 0.500 | 0.546 | +0.046 | 0.1819 | 0.1786 |
| ResNet-32 | 1751 | 66 | 3.8 | 0.045 | 0.136 | +0.090 | 0.1090 | 0.0511 |
| Transformer | 1671 | 110 | 6.6 | 0.248 | 0.433 | +0.185 | 0.0896 | 0.0633 |
| Pooled | 4534 | 495 | 10.9 | 0.260 | 0.454 | +0.194 | 0.1197 | 0.0868 |


Average precision (AP) summarizes precision over recall; higher is better. Brier is mean squared probability error; lower is better. Neither is an accuracy percentage or a causal effect. Prevalence helps interpret AP; AP should be compared between models on the same subset. Case rows subset pooled fits, with all three rates per case.


## D. Temporal variation of the primary comparison


| Test states | N | Positive | AP: loss | AP: + projected | Brier: loss | Brier: + projected |
| --- | --- | --- | --- | --- | --- | --- |
| 384-511 | 962 | 90 | 0.329 | 0.477 | 0.1290 | 0.1025 |
| 512-639 | 951 | 93 | 0.167 | 0.583 | 0.1700 | 0.0728 |
| 640-767 | 916 | 114 | 0.341 | 0.406 | 0.1183 | 0.1021 |
| 768-895 | 887 | 105 | 0.467 | 0.487 | 0.0923 | 0.0818 |
| 896-1016 | 818 | 93 | 0.473 | 0.525 | 0.0817 | 0.0732 |


These five ranges describe temporal heterogeneity, not five independent replications.


## Robustness: components and richer loss


All models here use a 24-state gap.


### Pooled


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.1146 | 0.09863 |
| loss_history | 0.2602 | 0.11972 |
| projected_change | 0.4543 | 0.08685 |
| raw_cosine | 0.4227 | 0.08923 |
| cosine_and_gradient_growth | 0.4328 | 0.09137 |
| rich_loss | 0.2580 | 0.12924 |
| rich_loss_projected | 0.4471 | 0.09360 |



### MLP


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.3571 | 0.25015 |
| loss_history | 0.5000 | 0.18186 |
| projected_change | 0.5457 | 0.17860 |
| raw_cosine | 0.5226 | 0.18540 |
| cosine_and_gradient_growth | 0.5417 | 0.18179 |
| rich_loss | 0.5323 | 0.18026 |
| rich_loss_projected | 0.5728 | 0.17295 |



### ResNet-32


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.0350 | 0.03758 |
| loss_history | 0.0454 | 0.10901 |
| projected_change | 0.1356 | 0.05108 |
| raw_cosine | 0.0946 | 0.05117 |
| cosine_and_gradient_growth | 0.0877 | 0.05936 |
| rich_loss | 0.0438 | 0.12128 |
| rich_loss_projected | 0.0862 | 0.06573 |



### Transformer


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.0632 | 0.06177 |
| loss_history | 0.2476 | 0.08959 |
| projected_change | 0.4327 | 0.06326 |
| raw_cosine | 0.3483 | 0.06512 |
| cosine_and_gradient_growth | 0.3985 | 0.06474 |
| rich_loss | 0.3021 | 0.10364 |
| rich_loss_projected | 0.4359 | 0.07000 |



## Robustness: past delay control


All models here use a 48-state gap.


### Pooled


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.1146 | 0.09877 |
| rich_loss | 0.2436 | 0.15112 |
| rich_loss_projected | 0.3815 | 0.11619 |
| rich_loss_projected_delay16 | 0.2475 | 0.14989 |
| rich_loss_projected_delay32 | 0.2542 | 0.14160 |



### MLP


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.3571 | 0.25070 |
| rich_loss | 0.5221 | 0.19254 |
| rich_loss_projected | 0.5506 | 0.18566 |
| rich_loss_projected_delay16 | 0.5232 | 0.19178 |
| rich_loss_projected_delay32 | 0.5188 | 0.19343 |



### ResNet-32


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.0350 | 0.03757 |
| rich_loss | 0.0400 | 0.15466 |
| rich_loss_projected | 0.0505 | 0.10397 |
| rich_loss_projected_delay16 | 0.0407 | 0.15629 |
| rich_loss_projected_delay32 | 0.0400 | 0.14419 |



### Transformer


| Features | AP | Brier |
| --- | --- | --- |
| training_prior | 0.0632 | 0.06181 |
| rich_loss | 0.3600 | 0.11986 |
| rich_loss_projected | 0.4762 | 0.08276 |
| rich_loss_projected_delay16 | 0.4051 | 0.11529 |
| rich_loss_projected_delay32 | 0.3572 | 0.10438 |



## Existing target/horizon sensitivity


| Horizon | Rise % | AP: loss | AP: + projected | AP: + all |
| --- | --- | --- | --- | --- |
| 4 | 5.0 | 0.478 | 0.565 | 0.707 |
| 4 | 10.0 | 0.316 | 0.470 | 0.598 |
| 4 | 25.0 | 0.096 | 0.326 | 0.470 |
| 8 | 5.0 | 0.390 | 0.561 | 0.569 |
| 8 | 10.0 | 0.260 | 0.454 | 0.395 |
| 8 | 25.0 | 0.193 | 0.365 | 0.248 |
| 16 | 5.0 | 0.438 | 0.546 | 0.528 |
| 16 | 10.0 | 0.384 | 0.461 | 0.441 |
| 16 | 25.0 | 0.345 | 0.396 | 0.383 |

## Full primary feature comparison (existing analysis)


| Case | Added features | AP | Brier |
| --- | --- | --- | --- |
| Pooled | loss_history | 0.2602 | 0.11972 |
| Pooled | projected_change | 0.4543 | 0.08685 |
| Pooled | gradient_norm | 0.3030 | 0.11152 |
| Pooled | momentum_norm | 0.2984 | 0.11467 |
| Pooled | all_diagnostics | 0.3953 | 0.10666 |
| MLP | loss_history | 0.5000 | 0.18186 |
| MLP | projected_change | 0.5457 | 0.17860 |
| MLP | gradient_norm | 0.5315 | 0.18070 |
| MLP | momentum_norm | 0.5637 | 0.17251 |
| MLP | all_diagnostics | 0.6253 | 0.16007 |
| ResNet-32 | loss_history | 0.0454 | 0.10901 |
| ResNet-32 | projected_change | 0.1356 | 0.05108 |
| ResNet-32 | gradient_norm | 0.0618 | 0.11038 |
| ResNet-32 | momentum_norm | 0.0526 | 0.11246 |
| ResNet-32 | all_diagnostics | 0.0631 | 0.11100 |
| Transformer | loss_history | 0.2476 | 0.08959 |
| Transformer | projected_change | 0.4327 | 0.06326 |
| Transformer | gradient_norm | 0.3881 | 0.06666 |
| Transformer | momentum_norm | 0.4321 | 0.07850 |
| Transformer | all_diagnostics | 0.5633 | 0.06658 |


Adding all diagnostics is not consistently best; preserve these results in the supplement.


## All fixed alarms, pooled forward-test states


| Alarm | Alarm states | True | False | Precision % | Recall % |
| --- | --- | --- | --- | --- | --- |
| eta_kappa_above_2 | 891 | 187 | 704 | 21.0 | 37.8 |
| eta_kappa_crosses_2 | 402 | 64 | 338 | 15.9 | 12.9 |
| gradient_grows_20_percent | 1008 | 175 | 833 | 17.4 | 35.4 |
| momentum_grows_20_percent | 612 | 158 | 454 | 25.8 | 31.9 |
| crossing_and_momentum_growth | 116 | 30 | 86 | 25.9 | 6.1 |
