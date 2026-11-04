# Analysis044: multi-spike intervention response

Completed2026-11-04. [Nine findings](FINDINGS.md) and [source-audited report](REPORT.md) compare the two approved policies with fresh same-worker controls across all three highest-LR cases. Mean loss changes for immediate half/cosine32 half are MLP-25.04%/-26.30%, ResNet+27.20%/+0.62%, Transformer-9.66%/-10.68%. Historical control reuse failed in every case; no historical bitwise replay is claimed.

[Immediate half](figures/fixed_half.pdf), [cosine32 half](figures/cosine32_half.pdf), and [combined overview](figures/all_results_preliminary.pdf) show the complete windows. Each comparison uses unchanged baseline, five raw metric rows, solid unsmoothed curves and independent scales. The Transformer cosine worker switch and exact replay are explicit. See [final figure observation](observations/006-final-results.md).

All235 transferred files and7 archive hashes pass verification;8,896 schedule/geometry checks and47,456 plotted values pass. All3 vector renders inspected. Run223 task Pods deleted and0GB task cloud storage retained. EstimatedUSD7.58 total, with provider billing still provisional. No manuscript change or push. Generate final figures/report with01_report.py;04_all_results_preview.py now delegates to final local reporting once production is closed.

Earlier partial observations and paired previews are retained as development history. The requested all_results_preliminary.pdf path now contains the completed combined figure; its former partial versions remain in Git history.
