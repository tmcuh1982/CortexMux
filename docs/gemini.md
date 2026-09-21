# Gemini through Google AI Studio

Gemini is an optional remote provider, disabled by default. CortexMux uses its
native REST API with the existing `httpx` dependency; no Google SDK or extra is
required. Enabling this provider sends the prompts you route to it to Google.
The default Ollama routes remain unchanged.

Create a key in [Google AI Studio](https://aistudio.google.com/apikey) and expose
it to the application as `GEMINI_API_KEY`. Do not commit the key to a TOML file.
For example, obtain it from your shell's secret manager before launching the
application. `api_key_env` can select another environment variable.

Save this configuration as `gemini.toml`:

```toml
[core]
allow_remote_hosts = false
approved_hosts = ["generativelanguage.googleapis.com"]

[providers.gemini]
enabled = true
api_key_env = "GEMINI_API_KEY"
timeout_seconds = 120
max_retries = 2
```

Set `CORTEXMUX_CONFIG` to that file's path. Enabling Gemini does not change task
defaults: select `provider="gemini"` explicitly, or configure a routing profile.
First inspect the available model IDs:

```python
from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    for model in mux.list_models("gemini"):
        print(model.name)
```

Discovery follows the model catalog's pages (at most 100), deduplicates IDs and
keeps models advertising `generateContent`. Use the returned bare ID, without
the `models/` prefix; no model is selected automatically. Availability and
structured-output support depend on the selected model and account.

```python
import os

from pydantic import BaseModel

from cortexmux import CortexMux


class Summary(BaseModel):
    summary: str
    item_count: int


# Set this to an ID from list_models("gemini").
model_id = os.environ["GEMINI_MODEL"]

with CortexMux.from_env() as mux:
    text = mux.generate(
        "Explain what this library does in one sentence.",
        provider="gemini",
        model=model_id,
        system="Be concise.",
        max_output_tokens=512,
        temperature=0.5,
    )
    print(text.content)

    result = mux.structured(
        "Summarize this list and count its items: apple, pear, orange.",
        provider="gemini",
        model=model_id,
        response_model=Summary,
    )
    print(result.parsed)
```

Use `async with CortexMux.from_env()` with `await mux.agenerate()`,
`await mux.achat()` or `await mux.astructured()` in asynchronous applications.
The context manager closes the provider's HTTP client. Each facade owns its
configuration and client independently.

Per-task model defaults belong under `[providers.gemini.defaults]` using the
keys `text_generation`, `chat` and `structured_output`. Empty strings mean
unset. Routing profiles accept `{ provider = "gemini", model = "..." }` for
these same tasks. Existing model availability validation also applies.

## Request behavior

The adapter accepts these options as keyword arguments on facade calls, or in
`options` on typed requests:

| Option | Accepted values |
| --- | --- |
| `max_output_tokens` | Positive integer |
| `temperature` | Finite number from 0 to 2 |
| `top_p` | Finite number from 0 to 1 |
| `top_k` | Positive integer |
| `stop_sequences` | List of up to five strings |

Unset options are omitted, preserving the selected model's defaults. Unknown
options are rejected before content generation. The existing `set_temperature`
API works with Gemini's declared 0–2 range. A model may impose additional
restrictions, which surface as a provider error.

For text and structured requests, `system` maps to `systemInstruction`. Chat
accepts leading system messages, user messages and assistant messages. Assistant
turns map to Gemini's `model` role; adjacent turns of the same role are grouped.
A conversation must begin and end with a user turn after leading system messages.
Images and tool messages are rejected.

`structured()` requests JSON output even without a schema. With a schema, the
adapter sends `responseJsonSchema` and validates returned JSON locally with
CortexMux's existing JSON Schema validator. `response_model` adds Pydantic
validation. The schema must be supported by both CortexMux and the selected
Gemini model; unsupported server-side constraints produce a provider error.

Blocked, empty, malformed and truncated responses raise typed errors. Only a
candidate completed with `STOP` becomes a successful response; thought parts
are excluded from returned text. Usage preserves Google's prompt, candidate and
total token counts. The total may also include thinking tokens, exposed as
`raw_metadata["thoughts_tokens"]`. Metadata records attempts and elapsed time;
the response model uses `modelVersion` when returned by Google.

The adapter currently supports non-streaming text, chat and JSON only. Vision,
embeddings, image generation, tool calling, grounding, thinking controls,
Gemini CLI and Vertex AI are not implemented.

## Transport and validation

The API key is sent in the `x-goog-api-key` header. Remote endpoints require
approval through the existing `core.approved_hosts` policy (or the broader
explicit `allow_remote_hosts` override). HTTPS is required outside loopback.
URL credentials, query strings and fragments are rejected; redirects are never
followed. The owned HTTP client ignores environment proxy settings. Errors
omit API response bodies, keys and request URLs.

HTTP 408, 429, 5xx, timeouts and transport errors retry at most `max_retries + 1`
times. The default is two retries with bounded exponential full-jitter backoff
(`retry_base_delay_seconds = 0.25`, `retry_max_delay_seconds = 2`). Authentication
errors, missing models and other non-transient statuses do not retry. A retry
can generate another billable request when the first response was lost.
`timeout` overrides `timeout_seconds` per HTTP attempt, not for the whole retry
sequence or the preceding model discovery.

Unit tests use `httpx.MockTransport`; no live Google calls or API key are needed.
Live model compatibility requires a separately enabled integration test.

API references: [generateContent](https://ai.google.dev/api/generate-content),
[model discovery](https://ai.google.dev/api/models), and
[structured output](https://ai.google.dev/gemini-api/docs/structured-output).
