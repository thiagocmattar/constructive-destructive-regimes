# Conventional intervention outcomes

- Changes are relative to each intervention's matched unchanged Adam control; negative loss changes are lower.
- One initialization; MLP states 513-800, ResNet 257-576, Transformer 545-928 for post-fork summaries.
- MLP uses half-squared error; ResNet and Transformer use cross entropy/NLL. Compare within each pair.
- Conventional and distributed-cap forks differ. This table is not a ranking across all four families.

| Family | Model | Setting | Mean loss change | Last-32 change | Maximum loss change | Test loss change | Clipping active |
|---|---|---|---:|---:|---:|---:|---:|
| beta2 | mlp | beta2=0.95 | +16.78% | +33.36% | +159.01% | -2.92% | n/a |
| beta2 | mlp | beta2=0.9 | +18.16% | +43.51% | +34.21% | -3.88% | n/a |
| beta2 | mlp | beta2=0.5 | +127.02% | +163.88% | +38.68% | -11.06% | n/a |
| beta2 | resnet | beta2=0.95 | +28.46% | +3636.05% | +407.18% | -32.14% | n/a |
| beta2 | resnet | beta2=0.9 | -3.53% | +981.27% | +71.08% | +2.46% | n/a |
| beta2 | resnet | beta2=0.5 | +6.28% | +506.38% | +0.01% | +11.85% | n/a |
| beta2 | transformer | beta2=0.95 | -13.15% | -13.15% | -54.46% | +24.64% | n/a |
| beta2 | transformer | beta2=0.9 | -18.47% | -21.21% | -61.57% | +33.12% | n/a |
| beta2 | transformer | beta2=0.5 | -11.82% | -14.92% | -64.79% | +27.77% | n/a |
| clipping | mlp | cap=1.0 | +48.01% | +376.50% | +281.67% | -7.75% | 43.4% |
| clipping | mlp | cap=0.5 | +2.08% | +24.48% | -15.32% | -3.12% | 82.6% |
| clipping | resnet | cap=1.0 | -28.32% | +78.03% | -52.15% | +8.32% | 99.4% |
| clipping | transformer | cap=1.0 | -17.74% | -18.68% | -71.13% | +14.84% | 63.3% |
| clipping | transformer | cap=0.5 | -19.02% | -16.33% | -71.13% | +11.30% | 90.9% |

- ResNet clipping at 0.5 remains missing; the [TODO](../../manuscript/consolidation/evidence/05-adam/c-intervention/TODO_ABLATIONS.md) records the remaining grid.
- [Per-arm CSV](conventional_summary.csv) includes absolute means, last 32 means, terminal losses and coverage; [results and checks](conventional_results.json) retain all source hashes.

## Unchanged-control context

- Loss-only excursions use the existing Analysis036 definition, including unrecovered events. Counts concern events with a post-fork peak, using the complete historical prefix for event boundaries.
- This is a descriptive portability check; no run is excluded or retuned based on the result.

| Family | Model | Excursions >10% | Excursions at least 2x | 2x peak within 32 states of historical target |
|---|---|---:|---:|---|
| beta2 | mlp | 16 | 4 | Yes (target 703) |
| beta2 | resnet | 4 | 2 | Yes (target 422) |
| beta2 | transformer | 5 | 3 | Yes (target 794) |
| clipping | mlp | 16 | 4 | Yes (target 703) |
| clipping | resnet | 4 | 3 | No (target 422) |
| clipping | transformer | 5 | 3 | Yes (target 794) |
