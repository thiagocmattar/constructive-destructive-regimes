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
