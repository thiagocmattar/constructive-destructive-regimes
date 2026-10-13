# Writing tone, wording and style

Based on the [manuscript draft](../draft/bare_jrnl.tex) and its sections,
cross-checked against the formal prose in [draft.md](draft.md). Informal notes
and TODOs are excluded.

- **Tone:** Formal, explanatory and mechanism-focused. Central claims are stated
  directly; interpretations are qualified with words such as "suggest,"
  "consistent with" and "can."
- **Voice:** First-person plural for the authors' actions: "we introduce,"
  "we derive" and "we observe." Gradients, updates and trajectories become the
  subjects when explaining how a mechanism works.
- **Wording:** Concrete verbs such as "align," "reinforce," "cancel,"
  "accumulate" and "overshoot." Technical terms remain consistent, including
  "cross-step interference," "consecutive-gradient cosine similarity" and
  "parameter displacement." Intuitive descriptions clarify the mathematics.
- **Argument:** Establish the existing perspective and its limitation, introduce
  the proposed formulation, then explain its practical consequences. Comparisons
  organize the discussion: local versus temporal behavior, constructive versus
  destructive interactions, and the effects of different optimizers.
- **Paragraphs:** Develop one main point through connected sentences: claim,
  explanation, implication. Transitions such as "Thus," "In particular,"
  "However" and "Therefore" make the logical relation explicit.
- **Mathematics:** Define symbols and assumptions before using them. Introduce
  each equation through the operation being performed, then explain its terms
  and meaning in prose. Build from plain SGD to optimizer extensions.
- **Results:** State the observed pattern, identify the relevant experimental
  conditions and figure, then interpret the mechanism. Separate exact identities,
  approximations, empirical observations and conjectures.
- **Presentation:** Descriptive section titles, equations and figure references
  carry the emphasis. The prose connects theory to observable training behavior
  with little rhetorical decoration.
