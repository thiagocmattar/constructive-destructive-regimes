# Run232: MLP Adam beta2 reductions

H039 / Analysis059. Authorized 7 November 2026: three new interventions on
owned Spark only, beta2 = 0.95, 0.9, 0.5 against the retained beta2=0.999 control.
Question: how does shorter second-moment memory change the selected MLP loss
excursions and training progress? No assumed direction of benefit.

Each branch loads exact Run223 state512, changes only native Adam's beta2,
and runs288 updates to state800. Preserve weights, m, v, global step count,
previous raw gradient and RNG; no moment reset/rescaling. LR0.001, beta1=.9,
epsilon1e-8, no clipping/decay, seed0. Full deterministic5000-image CIFAR-10
MSE objective, physical1000 partitions, original model/order and hashes.
Model/gradient/moments are FP32; whole-model diagnostics use FP64.
The reused control must pass exact fork gradient plus two-update replay of
weights, moments/clock, gradient history and RNG. Failure stops before treatments.

Native Adam recomputes bias correction as 1-beta2_new**(t+1), retaining the
global clock. This is an ordinary hyperparameter switch, not a product-of-past-
betas correction. It changes both the second-moment recurrence and the bias
correction at the fork. Record both denominators explicitly; do not attribute
effects exclusively to memory length. Executed scientific inputs remain frozen.

Every state512..800 records full loss, raw norm/cosine/projected change; every
outgoing update records LR/betas, moment/direction norms and adjacent cosines,
cos(g,m), cos(g,d), norm ratios, second-moment norm and correction, actual
increment norm, gradient-dot-increment and rounding residual. m/d at state t
are after incorporating g_t. Their first adjacent comparison uses the warm
state512 moments/direction from update511. Zero-denominator cosines/ratios
are null, without epsilon substitution. Adam's own epsilon stays1e-8.

Complete10000-image test evaluation at each endpoint is secondary. Primary
outcomes: mean training loss513..800, last32 mean, final and maximum loss,
>10% loss excursions with recovery/censoring from the established detector.
Compare each arm only with unchanged control; no tuning or additional seeds.
One selected development seed/window cannot establish generalization or
universal stability. Alignment sign is not finite-step loss decrease.

Save first-step audit513, rolling recovery every32 completed updates, final800,
optimizer/RNG/gradient and previous moment/direction state needed for resume.
Mechanical tests check beta placement, recurrence, native bias correction and
split replay; smoke computations are not added to scientific trajectories.

Compute plan: owned GB10, zero rental/cloud cost. Prior similar branches took
2-4 seconds each; allow10-60 seconds for three branches with extra diagnostics,
plus staging/verification. Hard worker timeout600 seconds and detached execution
survive SSH loss. Monitor within30 seconds; stop on nonfinite values or failed
qualification. Expected new artifacts below0.5GB, retained locally and on Spark
after SHA-256/size verification. Preserve all unrelated processes and files.

Reproduce: `01_worker.py` on the qualified Spark environment after staging the
hashed input manifest; `03_verify.py` locally after `02_remote.py collect`.
Analysis059 owns the report and figures. No manuscript edit or publication.

## Completed

All three288-update treatments completed7 November at12:48:50UTC. Exact
retained-control replay and added-observer replay pass;41 authenticated inputs
and41 retrieved artifacts pass SHA/size checks. First-step independent moment,
variance and adaptive-direction checks pass within FP32 rounding. The native
bias-correction transition is preserved explicitly; see `verification_final.json`.
The fork projected-change scalar differs by2.22e-16 because its squared-norm
reduction differs; unchanged tensor dynamics pass exact replay.

Worker12.15s, treatments3.59/3.83/3.77s;864 scientific updates and4 qualification
updates. Mean training loss increases16.78%/18.16%/127.02%; single-window results
and the three-column figure are in [Analysis059](../../analyses/059-2026-11-07-mlp-beta2-reduction/README.md).
Raw evidence and recovery/final/first-step checkpoints retained locally and on
Spark; no cloud resource or charge. Executed config/worker/test/inputs are frozen.
