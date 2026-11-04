# Analysis044: multi-spike intervention response

Run223 is in progress. MLP and ResNet results are verified; Transformer remains pending. Compare two approved policies with qualified historical or fresh same-worker controls across all three highest-LR cases. Five rows: logarithmic training loss, raw rho, raw gradient norm, eta kappa hat, applied/base LR. Solid unsmoothed curves, independent panel ranges, fixed windows and explicit fork/hardware provenance. Audit displaced/censored excursions and loss progress.

[Verified MLP preliminary figure](figures/mlp_preliminary_paired.pdf) and [seven scoped findings](observations/002-mlp-preliminary.md) are available; the Transformer case remains in progress.

## ResNet preliminary result

[Verified case-level findings](observations/003-resnet-preliminary.md): both arms completed and69 files verified; task Pod deletion confirmed. Mean loss versus fresh unchanged control:+27.20% fixed half,+0.62% cosine32 half. Transformer still pending; final three-model figures/report remain incomplete.

## Matching MLP and ResNet interventions

[Immediate half LR](figures/mlp_resnet_fixed_half.pdf) and [cosine32-to-half](figures/mlp_resnet_cosine32_half.pdf) each show MLP and ResNet side by side, paired with their own unchanged controls. [Comparison observation](observations/004-mlp-resnet-side-by-side.md) records sources, scope and caveats.125 artifact files and14752 plotted values verified; both vector renders inspected. Generate with03_mlp_resnet_preview.py. Transformer and the final three-model figures remain pending.

[Single-figure preliminary overview](figures/all_results_preliminary.pdf) places both interventions side by side for each model. MLP and ResNet are complete; Transformer control is complete through928; the11:28 BRT snapshot includes fixed-half through741 and cosine through586, with exact cross-worker replay verified. See observations/005-all-results-preliminary.md for coverage and provenance.
