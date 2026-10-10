# Models and data

Requested platform: Pythia-14M or larger Pythia models, trained/evaluated using
MiniPile. Exact model variant, initialization, revisions, sequence length, budget,
and preprocessing remain open for future runs. Run001 has selected and verified
the pinned source protocol below, with random initialization and complete tail
inclusion. Its config and input identities record the exact adopted choices.

Canonical references:
[Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) and
[MiniPile](https://huggingface.co/datasets/JeanKaddour/minipile).

## Reusable reference protocol, not an adopted default

The source study's `research/DATA.md` identifies:

| Input | Historical source identity |
| --- | --- |
| Dataset | `JeanKaddour/minipile` |
| Dataset revision | `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` |
| Tokenizer | `EleutherAI/pythia-14m-deduped` |
| Tokenizer revision | `7386d9a4ae45aef494a6e704910394def3037fc5` |
| Encoding | no added special tokens during encoding; EOS appended per document |
| Packed input length | 2,048 tokens |
| Source validation | all 500 validation documents |

At bootstrap these were references only. Run001 subsequently verified the exact
source token bytes and copied them to its task worker, recording 1,491,711,416
training input tokens and 693,668 validation input tokens. The run consumed both
short packed tails. No pretrained weights were imported. Future runs must verify
their own input identity rather than treating historical counts as measurements.

## Contract before preparation

- Resolve architecture/model/tokenizer variants and pin immutable revisions.
  Random initialization from architecture config is different from loading
  released pretrained weights; label continuation/fine-tuning explicitly.
- Define document/sequence/token sample units, EOS and packing policy, causal
  label shift, masks, supervised-token denominator, and leftover-tail handling.
- Record split names, source order, documents, tokens, complete blocks, excluded
  tails, token dtype, bytes, SHA-256, and preprocessing source/config identity.
- Match initial weights and data order across controlled conditions where
  appropriate. Record model/data seeds and an exact schedule identity.
- Declare validation and test coverage, any tuning split, and selection criteria
  before inspecting outcomes. Default comparisons should use identical coverage;
  a reduced smoke subset must be labeled.

Keep datasets under `data/` and derived caches under `cache/`, both ignored.
Reuse a source cache only after agreeing its protocol and verifying its bytes.
