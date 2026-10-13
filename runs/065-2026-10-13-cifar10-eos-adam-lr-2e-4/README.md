# Run065: CIFAR-10 full-batch ADAM, LR 0.0002

Complete:20,000 fixed updates on DGX Spark; full artifact retrieval, independent
scalar/eigenpair/state checks and final forward evaluation passed.
Final train loss/accuracy: 0.00070752871/100.00%;
final10k test loss/accuracy: 0.76159539/28.02%.

Owning [Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md)
defines the matched comparison and contains results, figures and interpretation.
[Reduced evidence](../../analyses/019-2026-10-13-cifar10-eos-lr/results/Run065.json) and [verification](../../analyses/019-2026-10-13-cifar10-eos-lr/verification/Run065.json)
cover all updates and retained final vectors. Immutable config and attempt source
bytes remain unchanged; scalar histories/checkpoints are retained outside Git.
