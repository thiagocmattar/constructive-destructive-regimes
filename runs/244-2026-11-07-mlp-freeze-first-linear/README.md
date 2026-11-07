# Run 244: freeze the first MLP linear layer

User-authorized H044 / Analysis064 causal intervention. At authenticated state448,
hold the first linear weights and biases fixed and continue native Adam through
state801. Display outgoing updates450-500 around the prior loss peak474 and
448-800 as a longer follow-up. State801 supplies the final outgoing displacement.
Do not infer that first-layer motion is the unique initiating cause from its
earliest forward appearance or from a successful freeze.

Matched: seed0, original initialization/data/order, all5000 CIFAR-10 training
examples, mean per-image half squared error, physical1000, tanh MLP3072-200-200-10,
Adam LR0.001, betas(0.9,0.999), eps1e-8, no clipping/decay, FP32 and TF32off.
Only applied increments to model[1].weight/bias are suppressed. Run native Adam
on all original raw gradients, then restore those two tensors before any next
forward. Thus their moments and clocks continue, LR is unchanged and later-layer
updates retain native semantics. Raw full gradients remain diagnostic quantities;
also report gradients restricted to the updated layers. No unfreezing is planned.

Qualify unchanged prefix0-448 against retained loss/gradient norms and native
diagnostics; compare full raw gradients and five representations at447/448.
Save fork448 and run a short qualification continuation to512, matching the
complete historical model, optimizer, RNG and previous/current gradients.
Restore fork448 for the intervention and verify all first-update later-layer
parameters and every Adam state against the qualified native update. At every
following boundary verify exact first-layer weight/bias and Z1/H1 invariance;
the fixed streams have zero displacement and undefined motion cosine.

Retain raw gradients and all five representation tensors447-501 for independent
event-window SVDs and reductions. Retain rolling full-data cosine, displacement
RMS, activation RMS, parameter/update/block norms, native gradient metrics and
loss through801. Save full fork, recovery and final checkpoints with optimizer,
RNG and gradient state. Native observer calls must preserve gradients and RNG.
Source/cache/code identities, progress logs, manifests and transfer inventory
are required. Frozen parameters comprise614600 of656810 parameters; this is
not a parameter-count-matched comparison to freezing another layer.

Owned Spark, no cloud charges. Previous unchanged replay took18s, but rolling
all-layer FP64 diagnostics and I/O add cost: expect below3min worker time;
600s internal cap and detached630s timeout plus30s TERM grace. Dense evidence is
about1.1GB, with raw plus archive below3GB, plus small source/recovery records.
Monitor read-only, retrieve and verify sizes/SHA256, confirm only this worker
exits, and preserve unrelated workloads. One seed, one fork, selected windows;
training behavior only, no generalization estimate. Executed inputs are immutable.

## Completion

Completed all353 frozen updates through801 in7.43s; replay, qualification and
diagnostics together14.80s. Peak allocated CUDA406.4MB. All54 source identities,
21 downloaded artifacts,449 exact prefix loss/norm pairs and complete historical
state512 checks pass. Both first-layer parameter tensors are bitwise unchanged;
all four later parameter tensors changed. All optimizer groups remain native
and all Adam clocks reach801. Dense/streaming reductions and zero-stream handling
are independently verified. See `verification.json` and Analysis064.

Output reversals450-500 fall51/51 to4/51 and maximum output displacement falls
87.38%; loss at474 falls46.43%. Extended mean loss448-800 is22.27% higher, despite
smaller spikes. One-seed causal evidence for a contribution of first-layer updates,
not proof of a unique initiating layer or improved generalization.

Task worker exited; `completion.json` confirms no remaining task process and
2,067,302,509 retained bytes on owned Spark, including archive/source records.
Transfer and local verification passed before closeout; zero cloud charge.
