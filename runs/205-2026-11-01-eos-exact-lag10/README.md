# Run205: exact checkpoint gradient lag profiles

Completed: 27 windows and 270 independently verified exact cosines. There are
21 exact saved starts, four qualifying reconstructed starts, and two unqualified
Run062 reconstructions retained separately. All 7.072 GB are local and hash
verified; all task cloud resources are deleted. Compute estimate USD 1.05,
provisional. See Analysis032 for results and figures.

Nine parent conditions from Analysis024, three anchors 500/2000/3000, ten plain
GD updates per window, and 11 complete-objective gradients per window. Raw FP32
vectors are retained; FP64 cosine reductions use no projection. Original source
and input identities, LRs, coverage and qualification policy are frozen in
`config.json` and `input_manifest.json`.

Seven conditions load exact saved model states. Two legacy MLP conditions need
initialization replay and must pass historical qualification for primary use.
Late windows extend beyond Analysis024's horizon and never modify parent files.
Scientific code: `01_worker.py`; mechanics tests: `02_test.py`. All raw evidence
belongs under ignored `artifacts/attempt-01/workerN/`.

See [Analysis032](../../analyses/032-2026-11-01-eos-exact-lag10/README.md) for
the measurement contract, execution limits, eventual results and reviewed plots.
