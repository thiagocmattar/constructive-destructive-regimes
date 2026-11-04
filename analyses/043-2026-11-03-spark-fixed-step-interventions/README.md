# Analysis043: fixed-step intervention responses on Spark

Completed all 36 matched branches across MLP/CIFAR-10, ResNet-32/CIFAR-10 and
Transformer/WikiText-2 at their highest tested Adam rates. Every branch starts
at exact warm state 1024 and performs 64 full-objective updates to state 1088.
The ten primary conditions and two optional cosine durations completed without
failure; both optional admissions satisfy the frozen time rule.

[Eight factual findings and full comparison](FINDINGS.md),
[report with all 13 vector figures](REPORT.md), [numeric results](results.json),
[terminal verification](verification.json), [storage accounting](storage.json),
and [source observations](observations/INDEX.md).

LR reductions lower MLP mean training loss about 45-47% here, while slowing
ResNet and Transformer training progress. Smoother raw gradient diagnostics do
not consistently rank attained loss. Double-median clipping never activates;
median clipping has mixed effects. Both decay strengths shrink parameter norms
without lowering mean/final training loss in this window. These findings are
conditional on one selected seed and warm state, not long-term convergence or
held-out benefit claims.

Main figures have three model columns, solid unsmoothed curves, independent Y
ranges, raw rho, gradient norms in row three and an explicit historical-to-Spark
hardware boundary. Decay companions are post-only because historical parameter
norms were not stored. All 34,326 plotted scalars match source arrays; every PDF
was rendered and visually inspected. Collection checks all 347 transferred files. The final verifier confirms
terminal inventory coverage, 31 frozen inputs, 19 executed source copies,
45 endpoint clocks/finite states and exact within-Spark replay. See
[Run222](../../runs/222-2026-11-03-spark-matched-interventions/README.md).

Run221's ResNet cross-GPU gradient gate failure (0.87% relative error versus a
0.1% gate) remains preserved. Fresh Spark controls and exact within-Spark replay
are the comparison basis; historical bitwise replay is not claimed. Shared before
states 960..1023 are context from prior hardware. The complete frozen design and
literature motivations remain in the [Run221 protocol](../../runs/221-2026-11-03-spark-fixed-step-interventions/PROTOCOL.md).

Reproduce from the repository root:

    python analyses/043-2026-11-03-spark-fixed-step-interventions/01_report.py
    python analyses/043-2026-11-03-spark-fixed-step-interventions/02_verify_final.py

The Spark worker exited at 06:52 BRT, before the 07:45 cutoff. Full evidence is
retained locally and remotely; the unrelated suspended queue is untouched.
USD 0 cloud spending, no manuscript edit and no push.
