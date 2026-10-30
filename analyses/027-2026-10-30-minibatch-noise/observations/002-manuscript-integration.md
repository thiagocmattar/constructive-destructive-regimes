# Observation002: manuscript integration and verification

Date: 2026-10-30. The approved design is implemented, executed and integrated
into the editable working draft. The supplied originals remain unchanged.

## Scientific organization

Section III retains all 16 equations. Two prose corrections distinguish the
nonpositive expected **unnormalized** noise projection from the normalized
random-denominator term, and distinguish a frozen/local cosine approximation
from a statement about whole-trajectory displacement.

Section IV's final transition now leads into sampling noise. New Section V,
*Mini-Batch Noise and Displacement Diagnostics*, presents the matched design,
frozen moment calibration, moving sampled/full-gradient geometry and the noisy
proxy interpretation. It contains two main figures and a generated paired
contrast table. Its results include approximation failures, initialization
variation, historical reconstruction failure and objective-progress counterexamples.

New Appendix E, *Additional Mini-Batch Diagnostics*, gives measurement coverage,
uncertainty, roundoff and new-reference replay checks, raw geometry, full loss,
and noise-corrected directional-HVP validation. A generated HVP error table
reports individual raw-pair medians/90th percentiles. The correction identity
is explicitly an accounting identity; independent HVPs assess approximation error.
The complete 5,000-example objective supplies all reference diagnostics.

The new section begins on page 9; its main figures/table are on pages 10–11.
Appendix E starts on page 19, with its evidence through page 22. References are
on page 23. Existing legacy results are retained; the introduction, conclusion
and older image-result interpretation still need the previously identified
broader editorial alignment. This task does not certify those older claims.

## Reproducible assets and checks

[07_build_manuscript_assets.py](../07_build_manuscript_assets.py) copies five
reviewed vector PDFs and two generated tables into the draft. The copied
assets and their source SHA-256 values are recorded in
[manuscript_assets.json](../manuscript_assets.json). The draft is self-contained.

[08_verify_manuscript.py](../08_verify_manuscript.py) independently recomputes
all 17 new table rows and 123 displayed scalars, checks 1,188 raw HVP pairs,
verifies all seven copied assets, and confirms unchanged Section III equations.
It checks the compiled PDF/log for the new section, unresolved references and
overfull boxes. [manuscript_verification.json](../manuscript_verification.json)
passes with 23 pages, zero unresolved references and zero overfull boxes.

All five figures were individually rendered and inspected, with embedded fonts
and unchanged plotting arrays. Affected manuscript pages were examined directly;
the remaining pages were checked in a contact sheet for reflow. Appendix floats
are flushed before References. The double-column float left a reduced available
column height; resetting it to the existing column height fixes the bibliography
without changing page margins. The IEEE reference trigger balances the current
list and is marked for revision if citations change. The final bibliography has
no large inter-item gaps or interleaved experimental figures.

The final PDF/source hashes and visual checks are in
[manuscript_review.json](../manuscript_review.json). Ignored build logs and renders
are under `manuscript/artifacts/draft/minibatch-noise-integration/`; the pre-edit
snapshot is retained there. Scientific input/raw evidence hashes are unchanged.
No additional training was needed for the editorial revision.
