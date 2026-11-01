# Run206: streaming exact long-lag continuations

Prepared for [Analysis033](../../analyses/033-2026-11-01-eos-long-lags/README.md),
following the user's request for far more lags on Runpod, as fast as practical,
with compression allowed only if needed. Streaming exact anchored products
remove the need for compression or a growing full-gradient history in memory.

`config.json`, `01_worker.py`, `input_manifest.json` and copied source protocol
metadata freeze the scientific contract. Sources are Run205's verified end
states at510/2010/3010 and its exact reference gradients at500/2000/3000.
All27 windows have common lag256 coverage, with declared extensions to2048
for persistent negative-lag1 profiles. All original eligibility flags persist.

`test_streaming.py` verifies independent FP64 reductions, undefined cases,
odd/even stopping, atomic raw retention and the unapplied-gradient resume boundary
on synthetic data. It is a mechanics check, not paper evidence. All new raw
artifacts, checkpoints and active logs live in ignored `artifacts/`.

The analysis README records matched/varying factors, full data coverage,
compute-chunk qualification, retention, limitations, live prices, bounded
execution and cleanup. The proposed USD30 larger envelope awaits approval;
no Runpod resources have been created for this run.
