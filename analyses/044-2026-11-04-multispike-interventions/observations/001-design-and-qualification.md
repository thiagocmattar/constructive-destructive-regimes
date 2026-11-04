# Approved multi-spike comparison

Source: Run223 config and authenticated Run215 histories. Seed0, highest LR; MLP512..800, ResNet256..576, Transformer544..928. Only fixed_half and cosine32_half. Pre-context64states. The frozen first large anchors are579/635/703,307/360/422,605/702/794.

Control reuse requires exact loss/norm observations through the prefix and8-step overlap, same GPU type and Torch/CUDA. This does not prove intermediate historical tensor identity; that limitation must remain visible. A failed gate requires a fresh same-worker control with the unchanged approved window. At least3 >10% events and a >=2x event near the historical selected peak are required before treatments. No retrospective window/seed search.

Raw rho and eta kappa hat describe adjacent gradients, not an Adam stability guarantee. Event counts are paired with mean, final and last32-state losses and censored/replacement events. Two figures compare unchanged with one treatment across3 model columns,5 rows, independent ranges, solid raw curves and fixed intervention markers.

Spark MLP completed, its historical exact-reuse gate failed, and its fresh control passed the frozen multi-spike gate. Both interventions and exact2-vs1+1 replay completed. Cloud outcomes pending. Partial findings are in partial_results.json and are not a completed three-case finding set.
