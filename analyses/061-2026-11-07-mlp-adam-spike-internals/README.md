# Analysis061: what changes inside the MLP at an Adam loss peak?

Run236 / H041. Exact replay of the unchanged Adam control around the last
pre512 local peak503 and recovery505, the containing excursion493, nearby
larger peak474, and secondary continued-control peak522. Full5000-example
internal measurements and selected checkpoint/activation snapshots; static
six-group parameter recombinations, output-change identities and actual-step
loss profiles. One seed and selected window; exploratory mechanistic analysis.

Primary question: parameter magnitude growth, activation saturation/change,
bias changes, and the role of the uncapped final linear layer. The user explicitly
authorizes diagnostic iteration on the owned Spark. No training intervention,
cloud resources or manuscript changes.

After the first measurements show nearly constant magnitudes and predominantly
hidden-feature contributions, `02_feature_probe.py` examines where activation
change energy occurs relative to saturation, how much lies in the fixed output
head's row space, and whether rise/recovery feature displacements reverse.
This is a declared post hoc diagnostic of retained Run236 snapshots; no new
optimizer updates or modification of executed Run236 inputs. The full5000
training examples and all200 hidden units remain included. Report exact norm
ratios, not per-coordinate RMS ratios as vector amplification.

## Completed findings

The evidence favors finite-step overshoot of output-sensitive hidden-feature
displacements. Parameter and hidden RMS magnitudes remain nearly unchanged
during individual spikes; restoring all tensor norms leaves about100% of their
excess loss. Hidden weight matrices receive98–100% of the static Shapley
allocation, while biases and the final head's own changes are small.

Tanh attenuates the observed feature changes. Almost all their energy occurs
in unsaturated entries; hidden layer2 has mean slope about0.843. Its feature
changes place64–69% of their energy in the head's row space, versus about4.6%
for baseline features. The head itself has spectral norm below one. Incoming
Adam updates initially point downhill but pass a lower-loss part of their
measured one-step paths. Recovery partially reverses feature motion while
weight norms continue their gradual growth. These are selected-window findings,
not a general stability threshold or unique causal attribution.

- [Full answer, tables, captions and limits](observations/001-spike-internals.md)
- [Internal time course](figures/01-internal-time-course.pdf)
- [Static mechanism probes](figures/02-mechanism-probes.pdf)
- [Numerical results and source hashes](results.json)
- [Independent probe verification](probe_verification.json)

`03_norm_direction_probe.py` adds post hoc static norm-restored/magnitude-only
networks, and `04_verify_probes.py` independently checks both probes on CPU.
Reproduce the figures with `python analyses/061-2026-11-07-mlp-adam-spike-internals/01_report.py`
using the retained local Run236 artifacts. All1,372 plotted points and both
vector PDF renders pass. All536 replay losses/norms are exact; state512 full
model/optimizer/RNG/gradient equality passes. No new control trajectory or
optimizer policy was adopted. Worker exited; about888MB retained on Spark
including archive, zero cloud cost. No manuscript changes.
