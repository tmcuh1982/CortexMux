# Grok through the xAI API

The `grok` provider supports text generation, chat and validated JSON in both
synchronous and asynchronous applications. It uses the existing `httpx`
dependency and requires no additional SDK. It is disabled by default; enabling
it sends explicitly routed prompts to xAI.

Create an API key in the [xAI console](https://console.x.ai/) and expose it to
your process as `XAI_API_KEY`, preferably through your secret manager. Do not
commit the key to configuration files. Save this configuration as `grok.toml`:

```toml
[core]
allow_remote_hosts = false
approved_hosts = ["api.x.ai"]

[providers.grok]
enabled = true
api_key_env = "XAI_API_KEY"
timeout_seconds = 120
max_retries = 2
```

Set `CORTEXMUX_CONFIG` to that file, or pass its path as
`CortexMux.from_env(config_path="grok.toml")`. When enabling Gemini as well,
include both `api.x.ai` and `generativelanguage.googleapis.com` in the existing
`approved_hosts` array. The built-in local routes remain unchanged.

List available text model IDs before choosing one:

```python
from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    for model in mux.list_models("grok"):
        print(model.name)
```

Discovery uses `/language-models`, includes advertised aliases, and excludes
entries declaring no text output. No model or alias is invented or selected
automatically. Access and supported parameters depend on your account and model.

Set `GROK_MODEL` to an ID from that listing, then use the existing facade:

```python
import os

from pydantic import BaseModel

from cortexmux import CortexMux


class Summary(BaseModel):
    summary: str
    item_count: int


model_id = os.environ["GROK_MODEL"]

with CortexMux.from_env() as mux:
    response = mux.generate(
        "Explain the purpose of a task router.",
        provider="grok",
        model=model_id,
        system="Be concise.",
        max_output_tokens=512,
    )
    print(response.content)

    result = mux.structured(
        "Summarize this list and count its items: apple, pear, orange.",
        provider="grok",
        model=model_id,
        response_model=Summary,
    )
    print(result.parsed)
```

`mux.chat(prompt="...", provider="grok", model=model_id)` supports conversations
through `messages` as well. System, user and assistant roles are preserved.
Images and tool messages are rejected. For async code, use
`async with CortexMux.from_env()` and `await mux.agenerate()`, `await mux.achat()`
or `await mux.astructured()`. Context managers close the owned HTTP client.

Per-task defaults belong under `[providers.grok.defaults]`, with keys
`text_generation`, `chat` and `structured_output`; empty strings mean unset.
Routing profiles accept `{ provider = "grok", model = "..." }` for these tasks.
The usual model availability check includes server-advertised aliases.

## Options and output validation

| Option | Accepted values |
| --- | --- |
| `max_output_tokens` | Positive integer |
| `temperature` | Finite number from 0 to 2 |
| `top_p` | Finite number from 0 to 1 |

Pass these as facade keyword arguments or through a typed request's `options`.
Unset settings retain xAI's defaults. Unknown options are rejected before
content generation. `set_temperature()` supports the adapter's 0–2 range;
individual models can impose additional restrictions, reported as typed errors.

Requests use `POST /responses` with separate role-tagged system and user input,
`store=false`, and no tools. Storage, web search, X search and code execution
cannot be enabled through request options.

`structured()` uses `text.format.type="json_schema"` with a schema, otherwise
`json_object` with an explicit JSON instruction. Results are parsed locally;
schemas use CortexMux's existing validator, and `response_model` additionally
validates with Pydantic. Schemas must also be accepted by the selected xAI model.

Only completed assistant text is returned. Incomplete responses, refusals,
unexpected tool calls and malformed outputs raise typed errors. Reasoning
content is excluded. Responses retain model identity and normalized token usage;
`raw_metadata` includes attempt count, duration, status and the response ID.

This adapter currently implements non-streaming text, chat and JSON. Streaming,
vision, image generation, embeddings, tool calling and reasoning controls are
not implemented.

## Transport

Keys are sent in the `Authorization: Bearer` header. HTTPS is required outside
loopback. Base URLs with credentials, query strings or fragments are rejected;
redirects are never followed. The owned client ignores environment proxy
settings. Error messages omit response bodies, credentials and request URLs.

HTTP 408, 429, 5xx, timeouts and connection errors retry at most
`max_retries + 1` times. The default is two retries, with bounded exponential
full-jitter backoff (`retry_base_delay_seconds=0.25`,
`retry_max_delay_seconds=2`). Other HTTP errors do not retry. A request-level
`timeout` overrides the configured timeout per attempt; it does not include
preceding model discovery. Retrying a lost response can incur another charge.

Tests use `httpx.MockTransport` and require neither an API key nor network access.
Live account/model compatibility must be verified separately.

API references: [Responses](https://docs.x.ai/developers/rest-api-reference/inference/responses),
[models](https://docs.x.ai/developers/rest-api-reference/inference/models), and
[structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs).
