# Run228: global gradient clipping1.0 on MLP

H035 / Analysis051. Authorized Spark-only pilot. Reuse the exact Run223 state512, its optimizer moments/clock, raw previous gradient and RNG; apply global L2 norm clipping at absolute threshold1.0 to the complete5000-image gradient before Adam, with constant base LR0.001. Run288 updates to800, keeping all other conditions fixed. No per-coordinate or adaptive-direction clipping; no moment reset.

Counterfactual: Run223 fresh same-Spark unchanged control, reused only after exact fork-gradient and two-update model/moment/history/RNG replay. Raw gradient diagnostics remain raw; log optimizer input norm and scale separately. Compare mean513..800, last32-state mean, endpoint and >10% excursions/recovery/censoring. Record clipped-update fraction, scale range and actual-step norms. Single seed/window; no mechanistic or generalization guarantee.

Budget: owned Spark only, zero cloud rental; recent matched branches2-5seconds plus staging/retrieval/verification, timeout1800seconds. Preserve unrelated suspended PID458832. Freeze source/config/inputs; no manuscript edit or push.

## Completed

Worker completed2026-11-04T21:20:40.632319Z.288 treatment updates;1.94seconds treatment,2.64seconds including qualification. Exact same-Spark replay passed and independently verified locally.125 updates clipped; LR remains0.001; final clocks800. [Analysis051 findings](../../analyses/051-2026-11-04-mlp-global-clipping/FINDINGS.md): mean loss+48.01%, maximum3.82x versus unchanged. No other models launched.

Collection verified17 new files and archive SHA; report additionally verified56 parent files. The first collector invocation failed because a copied hostname was accidentally changed to spark-2287.local. Restored spark-2277.local and collected successfully; training and executed science were unaffected. An auxiliary CPU verifier needed NumPy-array comparison and the correct completed_updates checkpoint field; the corrected verifier passed. Frozen worker/config unchanged.

Recovery/end checkpoints retained locally and on owned Spark. No cloud resources created, no rental cost. Unrelated suspended Spark PID458832 preserved. Local verification: test_clipping.py,03_verify.py, Analysis051/01_report.py. Plot source scalars, clipping rules, clocks and exact replay checked; vector render visually inspected.
