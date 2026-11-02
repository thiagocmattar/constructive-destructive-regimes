# Run212: exact backward anchored GD and Adam gradients

Owning analysis: [Analysis035](../../analyses/035-2026-11-02-eos-backward-anchors/README.md).
18 fixed256-update continuations from nine retained Run205 states, using the
largest source GD LR per model and existing calibrated Adam rates. Anchor at
the END of each window, t=start+256. Plot cos(g_t,g_(t-k)) at every k=1..256.
No moving adjacent-pair curve, forward-reference substitute, new rate tuning,
or compressed gradient projection. All source states directly saved/primary.

Executed inputs will be frozen in config.json/input_manifest.json. All257 FP32
gradients are temporarily stored on each worker, then reduced to exact endpoint
products with CPU/GPU checks. Six full audit vectors, every gradient hash, losses,
environment and terminal model/optimizer/RNG remain under ignored artifacts. Dense
histories are deleted after the complete products/audit vectors are saved. Raw gradients
are evaluated before their optimizer update; the endpoint gradient is unapplied.

Completed: 18 profiles / 4608 retrospective pairs; qualified starts, 90 local vector audits, all cosine/recovery/figure checks pass. All task compute/storage deleted; compute USD9.26, provisional. [Report](../../analyses/035-2026-11-02-eos-backward-anchors/REPORT.md).
