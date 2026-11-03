# Run217: local MLP Adam control foundation

Implementation/preparation only, 3 November 2026. H028 / Analysis039.
No neural training, smoke, timing calibration, intervention, remote execution,
spending or manuscript revision was performed. Coordinator preparation: `16a9f02`.
Run215 executed inputs remain unchanged. The 14-file input manifest pins this
new driver/config and 12 inherited source/input files, all matching Run215.

`01_control.py` is an executable A0 driver using `ProductionTask`, `model_for`
and ordinary `torch.optim.Adam.step()`. The fixed contract is the original
3072-200-200-10 tanh MLP, the same prepared 5,000 CIFAR-10 images in fixed
1,000-image chunks, half squared error summed over outputs/averaged over images,
frozen seed0 tensors, LR0.001, betas(0.9,0.999), eps1e-8, no decay/schedule/clip,
FP32 parameters/gradients/moments, FP64 products, TF32 off, foreach/fused false.
The current config provides A0 through state785; it implements no other arm.

## Observation clock and formulas

State t follows t complete updates. Before update t+1, `Observer.observe`
reads g_t, retained g_(t-1) and s_(t-1)=FP64(theta_t)-FP64(theta_(t-1)),
where both stored endpoints are FP32. It changes neither gradients, moments,
parameters, RNG nor observer history. It forms y_t=FP64(g_t)-FP64(g_(t-1))
and records all vector norms, s dot y, s dot g_previous, s dot s, adjacent raw
cosine, raw projected change `-g_previous.dot(y)/||g_previous||^2`,
completed-step `q_step=-s.dot(y)/s.dot(g_previous)`, and
`kappa_sec=s.dot(y)/s.dot(s)`. Signed nonzero denominators stay signed; zero
denominators and first-state history give null plus an explicit reason.
A zero numerator with a nonzero denominator gives a defined zero. No epsilon
replacement, sign test, spectral action or policy is implemented.

Under smooth exact full-objective gradients kappa_sec describes curvature
averaged along the completed step; FP32 endpoint gradients qualify that
interpretation. It does not identify current-direction curvature or predict
future loss. Raw projected change remains an Adam diagnostic.

`Observer.accept` runs after the outgoing ordinary Adam update. It delegates
four-stream capture to the unchanged `src/eos_tasks/adam_diagnostics.py` and
retains the current raw gradient, actual applied increment and complete scalar
pre/post history. Outgoing products and moment geometry occur only in
`metrics.jsonl`, with explicit post-update clocks. `boundaries.jsonl` contains
only pre-update observations and loss. Selected immutable vector audits retain
g_current, g_previous, y and s_previous at states0/1, selected snapshots and
endpoint; dense all-step history is not retained.

## Forks, recovery and identity

Immutable `checkpoints/main_fork-state-000657.pt` is published after657 updates,
before evaluating or consuming g_657. Further labels are pre_rise689,
onset690 and the requested endpoint. A sibling temporary file is flushed and
fsynced, then published using exclusive hard-link creation; an existing labeled
file raises instead of being overwritten. Only `recovery.pt` uses atomic
replacement. It is saved initially, every32 updates, on graceful boundary stop,
and at the endpoint. Completion retains both recovery and endpoint files.

Each checkpoint includes weights/buffers/training mode, all Adam moments/ages,
Python/NumPy/CPU Torch/initialized CUDA RNG, data order/position/normalization,
examples seen, preceding raw gradient and actual increment, all diagnostic
prefixes and scalar history, and scientific/runtime identity. Checkpoints are
complete-objective boundaries, with no pending current gradient; reload clears
`.grad` and recomputes that boundary once. Even endpoint checkpoints precede
endpoint-gradient RNG consumption. Resume requires a new output directory,
the same source/config/input/runtime identity and the same data/normalization.
Only trusted local checkpoint files may be deserialized.

`input_manifest.json` pins required source/config/prepared bytes; execution
verifies every hash before model construction and then checks prepared tensor
and initial-weight identities. No input is downloaded or renormalized.
`status.json`/`manifest.json` distinguish preparation, running, completed,
interrupted and failed execution, with exceptions/tracebacks. A failed or
partial update never replaces a good recovery. Last recovery may lag by up to
31 successful updates after a hard crash. No historical replay qualification
or neural reproducibility claim has been established.

## Unit verification and later execution

All tests use generated tensors/gradient mocks and pytest/system temporary
folders. The reused diagnostic unit test uses a generated miniature objective;
none invokes the production MLP or a training CLI. Final results and commands
are in `verification.json` and [Analysis039](../../analyses/039-2026-11-03-mlp-control-foundation/README.md).

Safe inspection command (does not construct a model):

```powershell
python runs/217-2026-11-03-mlp-adam-control-foundation/01_control.py --help
```

After a separate coordinator assignment and run-plan confirmation, the future
prefix command is `python runs/217-2026-11-03-mlp-adam-control-foundation/01_control.py
--output runs/217-2026-11-03-mlp-adam-control-foundation/artifacts/attempts/001-prefix
--until 657 --device cuda:0` (one shell line). This is documentation, not launch
authorization. Use a new attempt directory for a resume and pass `--resume`.
A smoke/calibration, held-out evaluation, replay qualification, intervention
implementation and all scientific conclusions remain subsequent assignments.
