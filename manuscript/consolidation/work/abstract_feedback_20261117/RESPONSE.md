# Abstract feedback: diagnosis and revision

17 November 2026. All three annotations in the supplied PDF are on the abstract. The abstract has been rewritten; the body and supplement are unchanged at source level.

## Why the previous version failed

The opening promoted one empirical observation into the paper's motivation. The phrase "these constructive and destructive regimes" then pointed back to opposition with decreasing loss, even though regimes are defined by the repeated signs of consecutive-update alignment. This could make loss behavior appear to be part of the definition. The next sentence introduced "secant" and a technical identity before making the paper's purpose clear. The abstract consequently read as a compressed method inventory rather than an argument connecting the four contributions.

## Responses to the three comments

| Annotation | Revision |
|---|---|
| Regimes are defined by reinforcement/opposition, independently of whether loss decreases. | Defined constructive/destructive regimes explicitly through persistent reinforcement/opposition between consecutive updates under plain SGD. Removed the ambiguous backward reference "these." Loss outcomes are reported separately. |
| Is opposition with falling loss really the main problem or gap? | Replaced that opening with the connection between optimization dynamics and network representations, and its implications for training control. The abstract then follows the framework, gradient diagnostics, functional measurements, and intervention in order. The original observation remains supported in the body without carrying the abstract's motivation. |
| Why "secant"? Reflect and rewrite the abstract. | Removed the term from the abstract and described the measurement directly: full gradients of a smooth, fixed objective recover directional curvature averaged along the applied step. Kept empirical Adam warnings distinct from that identity. Historical attribution and technical terminology remain in the body. |

The result statement retains the actual comparison: all tested cap quantiles improve mean training loss and accumulated upward variation relative to unchanged Adam in the three cases, while reversals persist. It does not claim that every endpoint improves or that training becomes faster. The final sentence returns to the intervention target.

## Files and checks

- [Revised manuscript](../../draft/manuscript.pdf): 20 pages, warning-free compilation.
- [Editable abstract](../../draft/sections/00-abstract.tex).
- [Preserved annotated original](../../draft/archive/2026-11-17/manuscript+feedback.pdf), copied byte-for-byte. The supplied file at the draft root is also untouched.
- [Extracted annotations](annotations.json) and [verification](verification.json).

The source-hash check permits changes only to the abstract, rebuilt main PDF, and draft README among the pre-existing files. All equations, body sources, numerical tables, figures, bibliography, and the supplementary PDF are unchanged. The first page was inspected at full render size, and all pages were inspected in contact sheets for downstream reflow. No unresolved references or overfull/underfull boxes remain.

To rebuild and verify from the repository root:

```powershell
python manuscript/consolidation/work/abstract_feedback_20261117/01_build.py manuscript
python manuscript/consolidation/work/abstract_feedback_20261117/02_verify.py
```

Build logs, pre-edit abstract, and renders are retained in ignored `scratch/` storage. No experiment was run.
