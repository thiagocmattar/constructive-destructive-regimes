# Observation003: complete pair interaction versus adjacent geometry

Question: does a history with predominantly negative adjacent cosine have
negative complete-history interaction, and is a lag-one approximation adequate?

Sources: all 10,000 recorded 2D vectors of every stream in Runs028-031;
verified NPZ identities in verification.json and figure_manifest.json. Retain
every step, no sketches/checkpoint thinning. Generator02_plot.py produces
figures/exact_temporal_interaction.pdf. It passed visual review.

Reduction: E_t=sum ||z_s||²; I_t=B_t²−E_t=2 sum_(i<j) dot(z_i,z_j);
L1_t=2 sum adjacent dot products; residual=I_t−L1_t. Prefix computation is
exact up to float64 arithmetic; independent small Gram matrices at early,
middle and final windows verify the identity. Cosine normalizes individual
pairs, whereas I/L1 weight their magnitudes; frequency of negative normalized
pairs cannot determine either aggregate's sign.

Caption: **Exact pair interaction versus lag-one truncation.** Raw g and
stored momentum m form columns. Rows show complete signed interaction I,
adjacent-only interaction L1, and the omitted residual I−L1. All four full
trajectories are plotted on independent linear Y scales; zero references
remain visible. Color/line styles match the cohort comparison.

Result: Run028 raw final E=294123.180862, I=2206559.830591,
L1=524568.936544, residual=1681990.894048. Complete-history interaction and
even the magnitude-weighted adjacent aggregate are positive, while 80.64%
of normalized adjacent cosines are negative. Large early gradients matter
to the aggregate; predominantly late cancellation does not erase their
contribution. Omitting lags>=2 omits the explicitly retained residual.
For stored momentum the analogous I/L1/residual are
2396027.872280/207340.269723/2188687.602557. These are separate quantities
from A²−B², whose sign is nonnegative by the triangle inequality.

Uncertainty/limits: the sums are exact for one finite deterministic history
per condition. Their signs and scales do not establish a stability result,
information destruction, generalization, or causal loss explanation. No
lag-decay assumption, smooth temporal decay, or universality is introduced.
