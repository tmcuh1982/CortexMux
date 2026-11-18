# Translation harness

This adapter uses the [generic engine validation harness](harness.md).

Use `CortexMux.translate(request)` or `await mux.atranslate(request)` to translate
one Chinese or Japanese field into French. Select two translators and an arbiter
from providers configured on the same facade. Each role needs structured output.
Set `think` to a reasoning mode supported by all selected deployments and their
profiles. For native oMLX set `think=False` and supply each engine’s bound
`behavior_profile` and `max_tokens`.

```python
from cortexmux import CortexMux
from cortexmux.translation import TranslationEngine, TranslationRequest

qwen = TranslationEngine(
    provider="omlx",
    model="mlx-community/Qwen3.8-27B-4bit",
    max_tokens=2048,
    behavior_profile="omlx-qwen-4bit@1",
)
gemma = TranslationEngine(
    provider="omlx",
    model="gemma-4-31b-it-4bit",
    max_tokens=2048,
    behavior_profile="omlx-gemma-4bit@1",
)
request = TranslationRequest(
    source_language="zh",
    text="20–25 公斤負載",
    citation="20–25 公斤負載，1750 公釐工作半徑",
    translators=(qwen, gemma),
    arbiter=gemma,
    think=False,
)
with CortexMux.from_env() as mux:
    result = mux.translate(request)
    if result.status == "accepted":
        print(result.translation, result.source_quote)
    else:
        print(result.reason)
```

Both translators independently receive the field, the citation and the same
translation policy. Python compares supported quantities including kg, mm,
file sizes, bounds, ranges and approximate durations. A multiset comparison
checks values and units. The arbiter checks what those measurements describe.
Quantities outside the recognized patterns can require review. This checker
covers a subset of technical language and does not certify all numerical claims.

The default glossary covers 喷涂, コストパフォーマンス and 分子公司. An explicit
`glossary` replaces those defaults. Pass known manufacturer names or product
references in `preserve` to require an exact copy when they appear in the field.

The arbiter receives eligible candidates under anonymous labels. With one
eligible candidate it judges that candidate alone. An accepted result contains
the selected engine, both final candidates in request order and a literal
source quote. The default `max_passes=2` permits one targeted recovery.
For an invalid vote or missing source quote, only the arbiter is called again.
For a correctable translation issue, each translator receives its own previous
draft and the first review reason. The revised candidates pass through the same
quantity, glossary, name and evidence checks. An unreadable or ambiguous source
quantity goes directly to review. If the second pass fails, the result remains
`review` and the selected translation is empty. A successful first pass stops
immediately. Use `max_passes=1` for the original single-pass behavior.
Inspect `passes_used` and `first_review_reason` to track recoveries. Provider/configuration errors
propagate to the caller. To validate a full record, call once for its value and
once for its full citation, accepting the record only if both fields pass.

Declare source identifiers such as `CB7` in `references` for exact copying and
exclusion from numeric parsing. Adjacent measures remain subject to the quantity
check. Use `decisive_span` to require the arbiter to cite a particular source
component. This checks the evidence span; semantic fidelity still requires judgment.
