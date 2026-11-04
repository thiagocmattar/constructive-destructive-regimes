# Run227: two-stage MLP cosine decay

H034 / Analysis050. User-authorized follow-up on Spark. From exact warm state512, decay base LR0.001 to0.00075 over32 steps, reaching the first floor at544. Hold until650, then decay to0.0005 over32 steps, reaching the second floor at682. Hold through799; observe endpoint800 without an update. No optimizer/moment reset.

Match Run223/226 seed0 MLP, full5000-image CIFAR-10 training objective, physical1000 partitions, frozen model/data/order/native Adam, FP32 vectors and FP64 diagnostics. Reuse Run223 fresh same-Spark unchanged and Run226 cosine32-to0.75x without rerunning controls. Exact fork gradient/two-update tensor replay, state544 full tensors, all boundary diagnostics512..651 and actual update rows512..650 must match the prior0.75x branch. The first step of the second cosine has unchanged LR, so the trajectories remain equal through651.

Record every raw-gradient boundary and actual applied LR/update. Retain complete model/moments/previous-current gradients/RNG at544,650,682 and800, plus recovery. Endpoint test evaluation covers all10000 images and is secondary. Compare mean651..800, last32, endpoint and >10% loss excursion amplitudes/recovery/censoring; also retain whole intervention-window means. Two paired figure columns compare the new branch with unchanged and with the existing held0.75x path. Solid raw curves and five diagnostic rows. One selected seed/window, no universal stability claim.

Compute plan:288 treatment updates plus2 qualification updates; recent branches2-4seconds, with several minutes for staging, transfer and verification. Spark only, no cloud rental. Task timeout1800seconds; preserve unrelated suspended PID458832 and ongoing Run223 cloud jobs. Freeze executed source/config/input hashes. No manuscript edit or push.

## Completed and verified

Completed2026-11-04 at12:05:37UTC on NVIDIA GB10. Worker2.67seconds including qualification; treatment1.95seconds. Exact fork-gradient/two-update replay, full state544 tensors,140 boundary rows through651 and139 update rows through650 match the previous0.75x path. All20 transferred artifacts and archive hashes verified; every applied LR, raw projected-gradient identity and checkpoint544/650/682/800 clocks/current gradients pass local checks. [Analysis050](../../analyses/050-2026-11-04-mlp-two-stage-decay/README.md) reports lower post650 mean loss and a smaller, shorter first excursion, with no >10% excursion after682. No cloud rental or manuscript edit.
