# Analysis020: full-batch and stochastic gradient interference across EOS tasks

**Current execution override (October15):** the user requested completion of
active Run075 and Analysis023 Runs120/121 only, then review before further
training. Remaining020 Runs076-103 stay held. Original configs and completed
evidence are preserved; the finish-only supervisor in Analysis023 supersedes
the earlier automatic queue continuation. No manuscript work is authorized now.

Status: the replicated frozen-state noise control and the three-case seed0
full-batch GD overview are complete, retrieved/hash verified and visually
reviewed. The [new 3x3 figure](figures/eos_gd_overview_seed0.pdf) uses one LR
per case and 500 updates, with loss, maximum curvature and directional curvature.
It is included after Table1 in the restatement report. See the
[observation](observations/002-gd-seed0-overview.md), [report](REPORT.md) and
[protocol](PROTOCOL.md). Active Run075 continues on Spark; the remaining
Transformer seed and SGD stage are held, as is the Analysis021 LR sweep.

The user selected CIFAR-10/tanh MLP, CIFAR-10/ResNet-32 and
WikiText-2/Transformer, with robustness appropriate for a technical paper aimed
at IEEE TNNLS. These are three task/architecture cases, using two real datasets.
The original EOS paper also studies VGG-11 and synthetic regression/deep-linear
tasks; it does not study ViT. See [source review](SOURCE_REVIEW.md).

Full-batch GD diagnostics compare unsmoothed objective loss, adjacent raw-gradient
cosine, the observed gradient-difference proxy eta*kappa, ordinary Hessian
sharpness eta*lambda_max, and complete-prefix normalized displacement B/A.
Mini-batch SGD will vary effective batch size with constant learning rate and
unchanged objectives. Multiple seeds and frozen-state noise controls will
separate stochastic cancellation from changes in the optimization trajectory.
The claim that noise lowers B/A and moves cosine towards zero is a hypothesis,
not an acceptance criterion.

ResNet uses the paper's fixed ghost-normalization groups of 1,000 images.
Sampling entire groups preserves its full-objective gradient in expectation;
changing BN group size would also change the objective. Language-model losses
are token weighted, with padding excluded and the complete prepared corpus
covered. Exact batch and diagnostic budgets follow measured calibration.

Run067 owns preparation, mathematical verification and calibration. Run068 owns
the completed control. Runs069-103 freeze the selected first-stage inputs;
Run061 supplies the verified matching MLP seed0 prefix. The user authorized
planning and execution on Spark; after an optional horizon question, the agent
selected the recommended bounded first stage and stated that assumption.
The longer paper-length horizon remains a subsequent scientific/budget choice.
Only the user-owned DGX Spark is authorized; no rented service is used.

The detached local controller checks terminal status every thirty minutes,
retrieves and verifies raw artifacts/checkpoints, and generates five publication
figures when all runs pass. It leaves visual acceptance and empirical manuscript
claims for review. Logs and active status are ignored; the remote queue survives
SSH disconnection. A numerical/diagnostic failure is retained and stops automatic
publication rather than silently dropping a condition.
The user reduced polling frequency on October15; Windows polling subprocesses
use `CREATE_NO_WINDOW` to suppress transient terminal windows. This changes
only local monitoring, not remote training, diagnostics or admission policy.

Publication figures: full-objective GD loss; GD cosine/projected change versus
sharpness; GD displacement/norms; SGD batch-size trajectories; SGD noise summary;
and the completed frozen-state control. GD cosine/projected-change panels show
100-pair trailing means with faint originals; loss, norms and B/A are unsmoothed.
All original scalar histories remain available.
