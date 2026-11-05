# Run230 | Combined cosine decay and global clipping on MLP

H037, Analysis053. Start from the exact Run223 state at 512 with seed 0, the 5,000-image CIFAR-10 objective, and warm Adam moments. Over the next 32 updates the cosine schedule takes the learning rate from 0.001 to 0.0005. Hold that floor through state 800. On every update cap the full gradient norm at 1 before Adam consumes it. Keep raw gradient history for diagnostics.

Use the original cosine and clipping implementations. Verify all 288 applied learning rates against the saved cosine-only arm and each optimizer input against the clipping rule. Reuse the three measured comparators after exact checkpoint qualification. Compare training-loss peaks, excursions, mean over 513–800, final 32 states and endpoint. Single seed and selected development window. Owned Spark only, expected worker time about three seconds, time limit 1,800 seconds, zero cloud rental.

Completed on the owned GB10 in 3.34 seconds for the treatment. All 17 new artifacts passed transfer verification. CPU read-back confirmed replay and all final Adam clocks at 800. [Analysis053](../../analyses/053-2026-11-05-mlp-cosine32-clip1/README.md) reports zero clipping activations and exact agreement with cosine alone. Recovery and terminal states are retained on Spark and locally.
