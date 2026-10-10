# Conclusion from the first schedule intervention

The schedule hypothesis makes sense and is supported in this first test:
accelerated decay reduces raw-gradient negativity and brings positive smoothed
alignment earlier at both 14M and 70M. Holding the midpoint LR prevents the
declared positive-recovery criterion at both sizes. The interpretation of that
geometric recovery needs to stay narrow: held LR reaches the lowest validation
loss, and accelerated decay the highest, at both sizes.

| Model | Schedule | Negative intensity D | Final validation loss |
| --- | --- | ---: | ---: |
| 14M | Original decay |0.0681 |5.1820 |
| 14M | Held midpoint LR |0.3620 |5.0674 |
| 14M | Accelerated decay |0.0222 |5.2176 |
| 70M | Original decay |0.1591 |4.0559 |
| 70M | Held midpoint LR |0.3201 |3.9846 |
| 70M | Accelerated decay |0.0981 |4.1042 |

`D=mean(max(-rho,0))` uses raw, unclipped adjacent-gradient cosines for
affected pairs 358-712. Full F/M/median statistics, recovery sensitivity,
source identities, coverage and figure captions are in
[observation001](observations/001-schedule-intervention.md).

Faster annealing reaches the 31-update smoothed-positive criterion 65 updates
earlier at 14M and 101 earlier at 70M. At 70M this recovery is temporary under both
decaying schedules. Negative alignment returns even after accelerated decay
has reached the LR floor. A sign transition is therefore not a demonstrated
convergence or stability transition.

The held schedule raises negative-pair frequency to 96.9% at 14M and 93.0% at 70M,
while lowering validation loss by 0.1146 and 0.0713 relative to original decay.
Accelerated decay lowers negative frequency to 11.3% and 51.5%, but worsens
validation by 0.0356 and 0.0483. The direction is consistent across both models;
the seed count is one. These results do not establish that negative alignment
causes better learning, or that one schedule will remain preferable over a
longer token budget.

Model-size ordering also depends on schedule. Unconditional negativity is
higher at 70M for original/accelerated decay, but higher at 14M for held LR in
this pilot. Keep Analysis003's replicated conclusions explicitly conditional
on its original recipe and time coverage. The dimensionality explanation for
model performance remains unidentified.

I would focus the central empirical claim on **schedule-sensitive local
gradient cancellation and its limited relation to loss**. A defensible wording
is: "In checkpoint-matched Pythia/MiniPile continuations, the learning-rate
schedule changes adjacent raw-gradient cancellation; increased cancellation
can coexist with lower validation loss." Replicate this result across seeds
before broadening it. LR-scaled AdamW decay and step scale change together;
their separate roles are not identified here.

Six parallel RTX5090 jobs completed 356 updates each from immutable update 356
checkpoints. All batches, states, byte inventories and complete validation
coverage were checked, and the new controls were compared with their archived
parents. GPU replay is numerically close rather than bitwise; the reported
schedule contrasts are larger than the observed aggregate control drift.

Cloud allocation-to-cleanup elapsed about 88 minutes. Estimated compute/storage
cost is USD 6.92; provider billing is provisional and currently incomplete.
All six workers and their volumes were removed after verification. Local
checkpoints/evidence are retained, unrelated cloud experiments are preserved,
and no multi-seed validation or manuscript rewrite was launched.
