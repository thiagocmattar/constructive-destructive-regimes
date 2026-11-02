Completed: [report](REPORT.md), [figures](figures/anchored_profiles.pdf), [exact adjacent curves](figures/adjacent_profiles.pdf).

Requested [largest-source-GD / Adam view, lags 1–256](figures/largest_lr_lag256.pdf)
uses the same nine-panel structure as Analysis033, with all consecutive lags
connected. Adam LRs: MLP 3e-4; ResNet-32 and Transformer 1e-3.
[Selection, caption and checks](observations/002-largest-lr-lag256.md).

The anchored figures here look forward from a fixed start, cos(g_s, g_(s+k)).
The backward endpoint definition, cos(g_t, g_(t-k)), has separate GD and Adam
figures in [Analysis035](../035-2026-11-02-eos-backward-anchors/README.md).

All task cloud deleted; USD 39.35 compute estimate, provisional.

# Analysis034: Adam continuations of the same EOS checkpoints

Authorized calibration followed by full Runpod execution. English records resolve
the dictated “Rampod” as Runpod and “taxas ... para o lado” as Adam learning rates.
Run209 calibrates one rate per model on the same 27 Run205 start states at
500/2000/3000. Run210 restarts cold Adam at those identical states and matches
each prior GD profile's measured horizon (256/512/2048). The historical
two unqualified Run062 reconstructions remain separate and never select rates.

Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8, fresh zero moments/step0,
constant LR, no clipping/decay/schedule/AMSGrad. Parent GD optimization histories
are matched inputs; Adam moments are not recoverable from those SGD histories.
The loss, full data/order, initialization, fixed ResNet BN groups, parameters,
FP32 gradient accumulation and FP64 geometry match Analysis033. Transformer
chunk512 must qualify against the original chunk64 at every start.

Every optimizer boundary measures exact raw-gradient norms, loss,
cos(g0,gk), and cos(g[k-1],g[k]). These are whole-model gradients along the Adam
trajectory, before Adam transformation. First adjacent cosine and zero norms are
undefined. No projection, clipping or random sampling is used. GPU FP64 products
are checked independently against CPU NumPy FP64 every lag. Retain g0/g1/g2,
the terminal pair, scalar history, model/Adam/RNG recovery and byte/SHA inventories.

Calibration: rates 1e-5/3e-5/1e-4/3e-4/1e-3, 64 updates each at all starts.
Choose one common LR per model using equal-primary-checkpoint mean
log(final16 mean loss / initial loss). Every primary state must be finite,
peak loss <=4 times initial and final16 mean <=1.05 times initial (plus1e-6).
Cosines never choose LR. This is short-horizon training-loss tuning on a declared
grid, not an optimal LR or independent optimizer-benefit evaluation.

Live preflight: 4090 USD0.89/h, 5090 USD1.19/h; up to19 independent GPUs,
USD50 aggregate ceiling, compute soft/hard guards47/49, 4h30 lease including
retrieval margin. Initial ETC2–4h is qualified from the previous workload and will
be refreshed by calibration.30GB container +15GB Pod volume per worker;
no network volume, all outputs retrieved/hash-verified before termination.
Task-owned resources only. Spark and unrelated work are outside scope.

No manuscript change, generalization, asymptotic damping or formal stability
claim. One initialization and dependent forks; uncertainty is descriptive.
