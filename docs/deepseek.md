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

The provider supports non-streaming text generation, chat, and JSON output.
Allowed options are `max_tokens`, `temperature`, and `top_p`. Structured output
uses JSON object mode and validates a supplied JSON Schema locally. It sends no
tools and does not expose reasoning text.

See DeepSeek's [context caching guide](https://api-docs.deepseek.com/guides/kv_cache/)
and [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/).
