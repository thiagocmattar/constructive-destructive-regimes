# Analysis035: fixed endpoints looking backward through 256 updates

Completed the requested separate [GD](figures/gd_backward_lag256.pdf) and
[Adam](figures/adam_backward_lag256.pdf) figures. Every curve fixes one endpoint
gradient g_t and plots cos(g_t,g_(t-k)) for k=1..256, in order. Source starts
500/2000/3000 become endpoints 756/2256/3256 after exactly 256 optimizer updates.
Rows are these later endpoints; columns are MLP, ResNet-32 and Transformer.
Each panel has one raw curve, common linear axes and no parity split or smoothing.

This implements the user's clarified backward definition. Historical
Analysis033/034 largest-rate figures were forward anchored cos(g_s,g_(s+k));
the adjacent figures were moving lag-one pairs. Neither former scalar series
can be reversed to obtain the new profile. The new endpoint's k=1 agrees with
its final adjacent pair, and k=256 agrees with its start-to-end cosine by
symmetry. Both identities pass in all 18 profiles.

Run212 computes 18 full-objective continuations from the same nine directly saved
Run205 states (all primary): largest source GD rates 0.04/0.1/0.2 from
Run122/120/126; Adam rates 0.0003/0.001/0.001 remain those calibrated in
Analysis034. Adam moments reset at the retained source, with beta1=0.9,
beta2=0.999, epsilon 1e-8, no clipping/decay/schedule/AMSGrad, foreach/fusedFalse.
GD is constant-rate plain SGD without momentum/decay. No new rate selection.

All source tensors and saved-gradient/chunk qualifications pass. Full data,
ordering, losses and parameter scope match the earlier contracts: MLP first 5000
CIFAR-10 images with half squared error averaged over images; ResNet all 50000
images with 50 fixed 1000-image train-mode BN groups; Transformer all 59680
sequences and 2088600 valid targets with token-weighted NLL. FP32 parameters and
accumulated raw gradients, TF32 disabled, exact FP64 dot products/norms.
Transformer chunk 512 qualifies against 64. Hardware and FP32 reduction differences
can change later trajectories; this is a recomputation from identical starts,
without a claim of bitwise equality to historical continuations.
The [historical scalar comparison](historical_comparison.json) records all 257
boundaries against the earlier forward profiles, including maximum loss,
relative-norm and forward-cosine differences; it never selects a result or horizon.
The observed maxima are 0.334186
in absolute training loss, 6.762987
in relative gradient norm, and 0.845000
in forward cosine. Later paths differ substantially in some cases. These figures
describe newly recomputed continuations from matched starts, rather than an
exact reproduction of the historical trajectories.

| Model | Start s | Fixed endpoint t | GD LR | Adam LR | GD rho_t(1) | Adam rho_t(1) | GD endpoint train loss | Adam endpoint train loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp | 500 | 756 | 0.04 | 0.0003 | -0.978879 | -0.994801 | 0.308877 | 0.127284 |
| mlp | 2000 | 2256 | 0.04 | 0.0003 | -0.980626 | -0.966043 | 0.193699 | 0.069397 |
| mlp | 3000 | 3256 | 0.04 | 0.0003 | -0.984743 | -0.797839 | 0.137786 | 0.040543 |
| resnet | 500 | 756 | 0.1 | 0.001 | -0.928536 | -0.204598 | 1.504890 | 0.508375 |
| resnet | 2000 | 2256 | 0.1 | 0.001 | -0.976083 | -0.403079 | 0.717812 | 0.117451 |
| resnet | 3000 | 3256 | 0.1 | 0.001 | -0.970753 | 0.986270 | 0.396903 | 0.008329 |
| transformer | 500 | 756 | 0.2 | 0.001 | -0.986877 | -0.601481 | 6.660555 | 2.343590 |
| transformer | 2000 | 2256 | 0.2 | 0.001 | -0.993872 | -0.588889 | 6.199159 | 2.239510 |
| transformer | 3000 | 3256 | 0.2 | 0.001 | -0.994295 | -0.556277 | 6.032049 | 2.169695 |

Validation covers 4626 gradients and 4608 backward pairs. Every pair has independent
GPU/CPU FP64 checks (maximum cosine difference 1.89e-15);
90 additional local pair audits use retained full vectors (maximum error
1.31e-14), and all 108 retained
vectors match their training-time hashes. Every plotted value matches the JSON,
CSV and artist arrays exactly. Both one-page vector PDFs pass structural and
actual-render visual review. [Checks](verification.json), [figure review](figure_review.json),
[all values](lag_profiles.csv), [results](results.json),
[observation/captions](observations/001-backward-anchors.md).

The temporary 257-gradient histories are reduced only after the endpoint is
available. Retain all raw pair products/norms, scalar histories and per-gradient
hashes, six full vectors per arm (offsets 0/128/192/254/255/256), terminal
model/optimizer/RNG recovery and source checkpoints. Dense histories are then
discarded on the worker to honor the request to simplify/accelerate execution.
No random projection or lossy gradient compression is used.

One seed-0 initialization per architecture and dependent checkpoint forks;
one pair per lag, without error bars or averaging endpoints. Optimizers change
both transformation and calibrated numerical step scale; actual update norms
are not matched. The figures do not establish permanent damping, stability,
generalization or optimizer superiority. No new validation/test or manuscript edit.

Five initial ResNet placements were retired before training: four failed the
saved-gradient threshold on RTX 5090, and one failed dependency setup. All five
were retried with unchanged scientific inputs on RTX 4090; failed logs and
verification receipts remain available. A transient FUSE-volume stall in one
Transformer arm cleared without an optimizer restart or scientific-input change.
The initial slow environment upload was replaced by direct official downloads
with source package versions pinned. [Execution notes](execution_notes.json).

All 18 task Pods and Pod storage were deleted after per-worker
byte/SHA-256 and archive verification: 312 files,
2.904 GB locally retained. No task network volume or
remaining cloud storage. Allocation through final deletion took 59.9 min;
compute estimate USD 9.26, provisional pending provider billing; USD 15 task
ceiling. Unrelated account volumes and Spark were preserved. [Closeout](closeout.json).

Generators: `03_analyze.py` (audits/figures), `04_records.py` (records).
Executed scientific inputs: Run212 `config.json`, `input_manifest.json`, `01_worker.py`.
