# Run059: CIFAR-10 full-batch GD, LR 0.004

Complete:20,000 fixed updates on DGX Spark; full artifact retrieval, independent
scalar/eigenpair/state checks and final forward evaluation passed.
Final train loss/accuracy: 0.096876419/98.94%;
final10k test loss/accuracy: 0.55870343/32.99%.

Owning [Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md)
defines the matched comparison and contains results, figures and interpretation.
[Reduced evidence](../../analyses/019-2026-10-13-cifar10-eos-lr/results/Run059.json) and [verification](../../analyses/019-2026-10-13-cifar10-eos-lr/verification/Run059.json)
cover all updates and retained final vectors. Immutable config and attempt source
bytes remain unchanged; scalar histories/checkpoints are retained outside Git.
