# Run219: temporary LR reduction using $\eta\widehat\kappa_t$

Completed local exploratory pilot; H028 / Analysis041. Approved plan5107aff:
seed0, same5,000-image CIFAR-10 full objective,3072-200-200-10 tanh MLP,
physical chunks1,000, original frozen weights/moments atstate657. Adam0.001,
betas(0.9,0.999), epsilon1e-8; FP32 parameters/gradients/moments, FP64 products;
TF32 off, foreach/fused false, no decay or pre-existing clipping.

Three conditions run367 canonical updates to1024: unchanged0.001, always-half
0.0005, and temporary0.0005 iff the available $\eta\widehat\kappa_t\geq1.8$,
otherwise0.001. Two absolute LRs; no compounding, smoothing, persistence or
threshold search. Finite undefined cases use base with a reason; nonfinite
training failures are preserved. At state s, manuscript t=s+1; the directly
computed raw-pair $\eta\widehat\kappa_t$ governs s to s+1 without LR rescaling.

Exact parent SHA256:
359097d026b45f8f0b8a33665aca4ad0382e52835745ceaeccf027ed29539ba4.
It is authenticated against unchanged Run217 source/config/input identity before
creating the new condition/source/horizon identity. Run218 executed inputs remain
unchanged. The new input manifest pins22 source/input files. Active manuscript
references are hash-recorded only and are not modified. Prepared parent files
are reused; no prefix training or dense all-step vector archive was added.

01_temporary_lr.py is a small continuation around existing model/observer/Adam
helpers. test_policy.py has four focused synthetic checks (4 passed,3.72s).
03_resume_control.py handles the concrete Windows JSON replacement lock using
bounded retries only; its synthetic two-failures-then-success check passes.

Attempt001-local completes always-half and temporary; unchanged hits WinError5
atstate900. Attempt002-control-recovery restores state881 with the same scientific
identity. Its19 repeated boundary observations match exactly. The original failed
status and traceback remain intact. Start/end/recovery and complete scalar
histories, optimizer ages, RNG and observer state are retained for every condition.

Canonical main budget:1,101 updates /5,505,000 image-update exposures. Actual
work includes16 qualification and19 replayed updates. Worker processes total
37.26s; start-to-recovered-finish218.34s, below the10-minute bound. Owned local
GPU only; zero cloud spending. No additional variant or full implementation
suite rerun. See completion.json and the two raw attempt manifests.

[Report](../../analyses/041-2026-11-03-mlp-temporary-lr/REPORT.md): the temporary
rule reduces LR18 times but yields worse excursion/progress outcomes than both
controls. Always-half remains strongest here. Fixed zoom673..745 and full view
657..1024 are retained without recentering. Single seed/selected state; no
special timing, robustness or generalization claim. No manuscript edit or push.
