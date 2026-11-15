# DeepSeek provider

The `deepseek` provider sends explicitly routed text requests to DeepSeek's
OpenAI-compatible Chat Completions API. It is disabled by default and remains
separate from a DeepSeek model served locally through Ollama.

Set the API key in `DEEPSEEK_API_KEY`, approve the provider host, and configure
the models used by each task:

```toml
[core]
approved_hosts = ["api.deepseek.com"]

[providers.deepseek]
enabled = true
base_url = "https://api.deepseek.com"

[providers.deepseek.defaults]
chat = "deepseek-flash"
text_generation = "deepseek-flash"
structured_output = "deepseek-flash"
```

Then select the provider explicitly or use the configured task default:

```python
from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    response = mux.generate(
        "Summarize this text in one sentence.",
        provider="deepseek",
        model="deepseek-flash",
    )
    print(response.content)
    print(response.usage.prompt_tokens)
    print(response.usage.completion_tokens)
    print(response.usage.total_tokens)
    print(response.usage.cache_hit_tokens)
    print(response.usage.cache_miss_tokens)
```

## Context caching

DeepSeek enables context caching automatically; CortexMux sends no cache switch
or cache key. Cache reuse depends on an exact shared prefix, so keep stable
system instructions, documents, and earlier conversation turns at the start of
the message list. Append changing questions or instructions after that prefix.
CortexMux preserves application-provided chat message order.

When returned by DeepSeek, normalized usage contains `cache_hit_tokens` and
`cache_miss_tokens`. The provider accepts DeepSeek's native
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` fields and the
OpenAI-compatible `prompt_tokens_details.cached_tokens` form. Cache population
and reuse are best effort, and entries expire according to DeepSeek's service
policy.

Every completed response also exposes the standard normalized token counters:

| Field | Meaning |
| --- | --- |
| `prompt_tokens` | Total input tokens, including cached and uncached input. |
| `completion_tokens` | Tokens generated in the model response. |
| `total_tokens` | Input and generated tokens combined. |
| `cache_hit_tokens` | Input tokens served from DeepSeek's context cache. |
| `cache_miss_tokens` | Input tokens that did not hit the context cache. |

DeepSeek defines `prompt_tokens` as the sum of cache-hit and cache-miss input
tokens when both cache counters are returned. A missing counter remains `None`;
CortexMux does not present an unavailable provider value as zero.

If DeepSeek returns usage with an incomplete response or invalid structured
output, CortexMux keeps the available normalized counters in the exception's
`context["usage"]`. Incomplete responses also expose `finish_reason` in the
exception context. The partial response text is excluded from diagnostics.

The provider supports non-streaming text generation, chat, and JSON output.
Allowed options are `max_tokens`, `temperature`, and `top_p`. Structured output
uses JSON object mode and validates a supplied JSON Schema locally. It sends no
tools and does not expose reasoning text.

See DeepSeek's [context caching guide](https://api-docs.deepseek.com/guides/kv_cache/)
and [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/).
