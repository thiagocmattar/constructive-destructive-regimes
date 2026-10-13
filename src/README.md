# Shared scientific code

Run002 reuses focused primitives from the executed Run001 in `baseline/`:
initialization, optimizer/scaler/attention recipe, token protocol, and streaming
vector geometry. Original Run001 sources remain frozen in its folder; reuse
provenance is recorded under Run002 `prelaunch/`. The geometry primitive now
also accepts explicit FP64 adaptive vectors. Run002's adaptive measurement is
reused in the subsequent matched runs; its archived executed source remains
unchanged. Runs 021–023 additionally share focused capture of actual Adam
`exp_avg` before bias correction/preconditioning, an optional warmup-only
schedule, and frozen-checkpoint initialization. Their source packet/inventory
preserve executed bytes independently of later shared-code changes. No general
training framework or sparsity package is introduced.

`cifar_eos/` promotes the frozen Run058 CIFAR-10 primitives for their second
use in Runs059-066: a small full-batch tanh MLP, GD/Adam, independent raw,
stored/bias-corrected momentum, adaptive-direction and actual-update geometry,
and residual-checked matrix-free ordinary/preconditioned Hessian spectra.
The production runner accepts an explicit run-local config and a fixed update
horizon. Run058 remains unchanged; Analysis019 inventories the executed source
bytes and retains an immutable copy in each Spark attempt. Its mathematical
verification covers loss reductions/accumulation, HVP/eigenpairs, optimizer
vectors, prefix identities and checkpoint continuity. No generic trainer is added.
