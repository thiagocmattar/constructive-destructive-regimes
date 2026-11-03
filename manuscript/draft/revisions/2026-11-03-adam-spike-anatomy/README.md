# Adam spike-anatomy integration, 2026-11-03

The author requested the following paper sequence: Adam equations and the role
of raw gradients; matched controlled training and loss overview; largest-spike
micro-plots; fixed-reference event definition and amplitude distribution;
grouped >10% anatomy table; pre-rise-cosine times rise-norm-ratio scatter;
interpretation and future interventions. This revision supersedes the previous
forecasting presentation in the active draft. Earlier evidence and review
snapshots remain preserved.

Deliverable: [revised-manuscript.pdf](revised-manuscript.pdf).
The active section is [06-adam-diagnostics.tex](../../sections/06-adam-diagnostics.tex).
The [evidence manifest](evidence_manifest.json), [verification](verification.json)
and [visual acceptance](review.json) record the numerical and layout checks.

## Implementation and scientific choices

- Copy the four author-selected Analysis036 PDFs byte-for-byte into the draft:
  loss-only overview, unsmoothed gradient/momentum spike views, peak-percentage
  KDEs, and the signed pre-rise-cosine/rise-norm-product regression scatter.
  The separate histogram alternative is not inserted because the author selected
  the density figure for the paper.
- Select the largest recovered relative peak at each highest tested LR, with
  no diagnostic or mid/late-time filter. Peaks are 703/422/794. In the existing
  micro-plots, gray shading is the +10% crossing to the loss peak, not the entire
  formal event. Windows and normalization remain exactly as supplied.
- Define first strict departure from fixed pre-rise loss, first return, earliest
  loss maximum, ratios and durations. Do not split partial recoveries into new
  events. The full density sample has 806 recovered excursions and no amplitude
  cutoff; the >10% anatomy sample requires observed recovery and has 129 events.
  Two above-threshold candidates without observed recovery are excluded, as
  requested. The 10% cutoff is an exploratory operational choice; the density
  does not establish a natural separation or a stability threshold.
- Replace bracketed pairs with grouped table headers/subcolumns. Peak mean/SD
  and maximum have separate columns; rise/recovery times, before/recovery
  diagnostics and rise/recovery norm ratios each have their own subcolumns.
  Peak changes and times display one decimal; other statistics display two.
  Before is a single state-a observation, recovery diagnostics are event means,
  and norm ratios use endpoints. Every original value is preserved before rounding.
- Distinguish the three illustrative negative-cosine starts from the population:
  117/127 defined pre-rise cosines are negative; all 129 peak-state gradient norms
  exceed their pre-rise norm. Above-two projected change is present before the
  first loss increase in 21/127, not all events. The Transformer crossing at 788
  precedes its +10% crossing and loss maximum but is already at the first
  elevated loss state.
  The selected ResNet/Transformer constructive recovery segments are not asserted
  for all events; population recovery averages remain model-dependent.
- Treat the scatter product as a retrospective event descriptor, distinct from
  the consecutive-gradient projected-change identity. R-squared is fitted and
  evaluated on log10 peak/pre-rise loss within each LR. Dashed extensions are
  extrapolations. The result supports an association, without a new stability
  theorem, prospective detector, causal feedback claim or intervention benefit.
- Update the abstract, introduction, conclusion and Adam measurement appendix
  to match the current evidence. Remove the old forecast table/protocol from the
  compiled presentation; keep historical files and review originals.
- Cite gradient clipping as motivation for future tests, with its Adam moment
  effects stated explicitly. Literature checked against primary sources:
  [Pascanu et al., ICML 2013](https://proceedings.mlr.press/v28/pascanu13.html)
  and [Zhang et al., ICLR 2020](https://arxiv.org/abs/1905.11881).
  No claim transfers their theoretical conditions to the measured Adam runs.

## Reproduction and checks

From the repository root:

```text
python manuscript/draft/revisions/2026-11-03-adam-spike-anatomy/01_prepare_assets.py
python manuscript/draft/revisions/2026-11-03-adam-spike-anatomy/02_build_and_verify.py
```

The first script verifies hashes, independently segments all 806 recovered events,
checks the 129 event anchors and recovery means against raw histories, recomputes
table statistics, checks example selection/timing and regression values, and copies
the approved assets. The second retains the preceding GD/SGD evidence and 186
mathematical checks, checks the new table/section integration, compiles with
pdflatex/BibTeX, checks all nine active PDF inputs and renders every page with Poppler.
Build/render scratch is ignored under `tmp/pdfs/`; the review PDF lives here.
Final visual review and PDF hash are recorded separately after rendering.

No training, cloud resources, hypotheses or new interventions are introduced.
Original Analysis036 figures and event records, the supplied manuscript/review
originals, and the prior review snapshots are unchanged. This is a descriptive
single-initialization analysis with repeated, potentially dependent events.
