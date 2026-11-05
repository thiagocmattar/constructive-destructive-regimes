# Observation003: independent verification and visual review

The final [verification receipt](../verification.json) records a passing run of
[03_verify.py](../03_verify.py), with pinned verifier/reduction/plot hashes.
The verifier reads the original logs independently and does not invoke the
analysis functions to regenerate its expected reductions.

| Check | Coverage/result |
| --- | --- |
| Source/protocol and output SHA-256 | All pinned sources and six derived tables; no source mutation |
| Raw states, loss/norm/log features, clock and Q identity | All 9,216 prospective state rows |
| Independent streaming fixed-reference segmentation | All 812 candidates |
| Historical future labels/eligibility | All 4,534 archived forecast rows |
| Eligible full-history decisions | 7,204 with 645 future-rise markers |
| Maximum matching cardinality | All 18 cases, independent sparse-graph algorithm |
| Pair membership, unique use and summary aggregation | 764 pair records across two contrasts |
| Pair calipers | 4,584 constraints |
| Context balance before/after | Every exported feature/group record |
| Numeric figure data | 99 records /46,098 points |
| PDF format | Five single-page,7.16-inch-wide vector PDFs, embedded fonts, no raster images |

Every PDF was rendered with Poppler at 2,000-pixel width and visually inspected.
The middle-rate figure additionally received a full-page PyMuPDF render to
check its complete layout. All panels, titles, ticks, axis labels, gray bands and
legends are present and readable, with no text overlap or page clipping.
Dense cosine traces intentionally retain every unsmoothed observation; vector
PDF zoom resolves individual segments. Common geometry axes include every point.

Figure-specific review:

- Low-rate timelines: readable log ticks, quiet control paths and dense reversal
  are preserved; empty >10% bands in ResNet/Transformer reflect the data.
- Middle-rate timelines: all three headers, loss/norm ranges and full0..1024
  histories visible; no startup crop or hidden extreme.
- High-rate timelines: later constructive ResNet and recurring MLP/Transformer
  events visible; censored rise outlines differ from recovered gray bands.
- Joint map: colors and shapes distinguish outcomes; eligibility counts,
  GD-only reference and the four-entry legend clear the panel axes.
- Matched contexts: group symbols, learning rates, n counts and sparse flags
  are legible. Empty/tied future-rise groups remain at zero; no CI implied.

The legacy Analysis036 verifier's known README hash mismatch is not repaired or
used to waive a data check. This verifier independently audits the underlying
source hashes and all needed event/forecast values. Prior scientific manifests,
winning/source figures and manuscript files remain unchanged. No new scientific
trajectory or failed source test is concealed by documentation normalization.
