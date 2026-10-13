# Run066: CIFAR-10 full-batch ADAM, LR 0.0004

Complete:20,000 fixed updates on DGX Spark; full artifact retrieval, independent
scalar/eigenpair/state checks and final forward evaluation passed.
Final train loss/accuracy: 0.0006593872/100.00%;
final10k test loss/accuracy: 0.70923975/28.35%.

Owning [Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md)
defines the matched comparison and contains results, figures and interpretation.
[Reduced evidence](../../analyses/019-2026-10-13-cifar10-eos-lr/results/Run066.json) and [verification](../../analyses/019-2026-10-13-cifar10-eos-lr/verification/Run066.json)
cover all updates and retained final vectors. Immutable config and attempt source
bytes remain unchanged; scalar histories/checkpoints are retained outside Git.
