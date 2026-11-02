# Run216: Transformer accumulation at physical microbatch 512

H023 / Analysis037. The user approved US$40 total / four hours and requested
continued GPU searches for parallel execution. Run214 has begun its ResNet
measurement and remains frozen. This separate run changes Transformer physical
accumulation from 64 to 512, selected by mechanics-only timing before production.
The user subsequently extended the shared wall limit to six hours, keeping
US$40 total; the hard deadline is 2026-11-02 19:34:56 UTC.

Complete-objective calibration on RTX5090 reduced a full gradient from about
5.3 to 4.1 seconds. At the same pinned initial tensors, the relative gradient L 2
difference was1.53e-6, cosine agreed within1e-8 and loss within1e-5; peak memory
was 10.33 GB. The objective and effective batches are unchanged. Three fresh
2000-update GD references and all continuations use this runtime/accumulation.

Reuse the frozen Run214 measurement implementation via explicit thin wrappers.
Its bytes, this config and the prepared data/initial tensors are in the input
manifest. Retain three seeds, states 0/2000, batch sizes 8/32/128/512/59680, eight
stochastic repetitions, one full control and 200 updates per fork. Plain SGD
LR 0.2, FP32 gradients, FP64 geometry and TF32 disabled. No result-based tuning.

The existing Analysis027 MLP and Run214 ResNet arms complete the three-case
comparison. Analysis037 owns the joint reducer and publication figure. All Pods,
calibrations, retries and storage share the same approved envelope; no additional
spending authority is inferred. Raw evidence and active logs stay ignored.

Complete:198 continuations and three 2,000-update references pass exact raw
inventory, sampling/geometry and endpoint-displacement checks. The initial
seed 0 full control reuses its exact reference prefix, giving 45,400 executed
production updates. All task cloud resources are deleted. Joint Analysis037
acceptance, publication figure and manuscript review pass; total task cost
US$26.30 provisional across both runs, including retries/calibration/storage.
