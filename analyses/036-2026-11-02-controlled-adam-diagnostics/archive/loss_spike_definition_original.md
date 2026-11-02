I would define a loss spike as a maximal upward excursion from a reference loss level, rather than simply as a local maximum.

The important distinction is that a local maximum identifies one timestamp, whereas an excursion gives us three timestamps:

\\[ \boxed{(t\_{\mathrm{onset}},\ t\_{\mathrm{peak}},\ t\_{\mathrm{recovery}})} \\]

These are sufficient to characterize the rise, maximum excursion, and recovery of the spike.

## 1. Mathematical definition

Let \\(\ell_t\\) denote the observed training loss at optimization step \\(t\\).

Because the loss generally decreases during training, let \\(b_t\\) denote a reference loss trajectory (a slowly varying local baseline). Define the excess loss

\\[ x_t = \ell_t-b_t. \\]

Let \\(\epsilon\geq 0\\) be a tolerance around the baseline.

Definition (Loss excursion). A loss excursion is a maximal interval \\((a_i,c_i)\\) satisfying

\\[ \boxed{ \begin{aligned} x\_{a_i}&\leq\epsilon,\\\ x_t&>\epsilon,\qquad a_i\<t\<c_i,\\\ x\_{c_i}&\leq\epsilon. \end{aligned} } \\]

In other words, an excursion begins when the loss crosses above the reference level and ends at its first subsequent return to that level.

Definition (Peak). The peak of excursion \\(i\\) occurs at

\\[ \boxed{ p_i=\underset{a_i\<t\<c_i}{\operatorname{argmax}}\ x_t } \\]

with the earliest maximizing timestamp selected in the event of ties.

We can optionally require a minimum excursion amplitude \\(h>\epsilon\\) to distinguish significant spikes from ordinary fluctuations:

\\[ \boxed{x\_{p_i}\geq h.} \\]

However, I would initially retain every excursion as a candidate and apply the amplitude threshold afterward.

## 2. Geometric interpretation

Illustrative excursion with onset \\(a\\), maximum \\(p\\), and first recovery \\(c\\). The horizontal reference is drawn constant for clarity.

This gives natural, unambiguous quantities:

\\[ \begin{aligned} T\_{\mathrm{rise}} &= p_i-a_i,\\\\[3pt] T\_{\mathrm{recovery}} &= c_i-p_i,\\\\[3pt] A_i &= x\_{p_i}-\epsilon. \end{aligned} \\]

The excursion need not rise or fall monotonically. Small fluctuations inside the excursion do not change its boundaries.

## 3. Returning to the pre-spike loss

Your specific suggestion is that the spike should end when the loss returns to its pre-spike value.

If we use the observed loss at onset as the reference, this can be defined using a first-passage time:

\\[ \boxed{ c_i^{\mathrm{raw}} = \inf\left\\{ t>p_i: \ell_t\leq\ell\_{a_i}+\epsilon \right\\}. } \\]

This is slightly different from recovering to the estimated trend \\(b_t\\).

For a neural network, I would favor recovery to \\(b_t\\), because the baseline training loss can decrease significantly during a long spike. Nevertheless, the raw-loss criterion is perfectly valid if literal recovery to the pre-spike value is what you want to measure.

## 4. Two important edge cases

Multiple local maxima within one excursion. Under this definition, they belong to the same spike, with the largest maximum designated as its peak. Separating them would require another criterion, such as minimum peak prominence.

No recovery before training ends. If \\(x_t\\) never returns below \\(\epsilon\\), the excursion is right-censored. Its recovery duration is undefined rather than zero or the remaining number of training steps.

## My recommendation for the paper

I would define the fundamental object as an excursion triplet \\((a_i,p_i,c_i)\\), with the peak identified by its maximum excess loss, and retain the magnitude threshold \\(h\\) as a separate selection criterion.

The one conceptual question worth settling is what constitutes the pre-spike baseline:

1. The observed loss immediately before the excursion.
2. A local estimate of the typical loss trajectory.

I favor (2) for large-scale neural network training, because it accommodates a changing loss trend. Once that baseline is specified, the onset, peak, and recovery can all be defined precisely without requiring the ascent or descent to be monotonic.