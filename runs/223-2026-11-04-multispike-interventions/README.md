# Run223: multi-spike fixed-step interventions

Authorized 2026-11-04. Running; three-case findings remain pending. H030 / Analysis044.

Only fixed half LR and cosine32 to half, plus a contemporary control only when historical reuse cannot be qualified. Windows: MLP512..800, ResNet256..576, Transformer544..928. Seed0, original full objectives and physical partitions. Shared warm forks retain Adam clocks and raw-gradient history.

Cloud cap USD15 including setup, failed provisioning, compute and temporary storage. At most3 simultaneous task GPUs. Spark runs MLP; cloud runs ResNet and Transformer. Prefer same-type workers for the Transformer split, with exact cross-worker qualification. No extra variants/seeds, manuscript changes or push. Preserve unrelated Spark PID458832 and preexisting Pods.

Soft spending limit12, compute stop14, final ceiling15; stop task processes and retrieve hash-verified artifacts before deletion. Guard is independent of orchestration. See config.json for frozen qualification and event gates. Estimated3-4h conditional on availability and measured throughput; refresh after real calibration.

## Launch verification

Spark MLP completed both treatments; its historical trace did not reproduce exactly, so a fresh same-Spark control was used and passed the predeclared event gate. Exact2-vs1+1 checkpoint replay passed. All56 Spark artifact files and transfer archive were hash-verified. The ResNet leader is running on RTX4090 after a3.40s/update calibration. Transformer capacity/calibration is managed in scratch/production-lease.json. Failed initial metadata loading and Spark transport/environment attempts are preserved in infrastructure_corrections.json; no accepted cloud training update preceded the metadata correction.

Detached controller launch/guard identities are in controller_launch.json and monitor.json. Heartbeat run223-multi-spike-interventions follows every10minutes. The controller retains the Transformer leader until the follower passes exact replay. Its per-leader admission ceiling is USD9 projected Transformer compute (includes1.2x timing margin); aggregate guard remains12/14/15 and includes all setup failures.

## ResNet completion (2026-11-04)

Both ResNet interventions completed at12:09:41UTC. The controller collected69 size/SHA-verified artifact files before deleting task Podduqoihms8w1gle; independent recheck and providernot_found404 confirm retrieval/deletion. Historical trace reuse failed, so comparisons use the qualified fresh same-worker control. Exact resume replay and the event gate pass. Mean loss is27.20% higher for fixed-half and0.62% higher for cosine32-half than unchanged; endpoint/tail training loss also favors unchanged. See resnet_completion.json and Analysis044 observations/003-resnet-preliminary.md. Transformer remains active; no three-model conclusion yet.
