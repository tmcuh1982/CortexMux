# Qwen Cloud provider

The `qwencloud` provider sends explicitly routed text requests to Qwen Cloud's
OpenAI-compatible Chat Completions API. It is separate from `ollama`: a Qwen
model served locally through Ollama keeps `provider="ollama"`.

Qwen Cloud is disabled by default. Configure the API key in `DASHSCOPE_API_KEY`
and approve the exact provider host before enabling it:

```toml
[core]
approved_hosts = ["maas.qwencloudapi.com"]

[providers.qwencloud]
enabled = true
base_url = "https://maas.qwencloudapi.com/compatible-mode/v1"

[providers.qwencloud.defaults]
chat = "qwen3.8-flash"
text_generation = "qwen3.8-flash"
structured_output = "qwen3.8-flash"
```

Then select the provider and model explicitly:

```python
from cortexmux import CortexMux

with CortexMux.from_env() as mux:
    response = mux.generate(
        "Summarize this text in one sentence.",
        provider="qwencloud",
        model="qwen3.8-flash",
    )
    print(response.content)
```

To use Qwen 3.5 Flash on the same pay-as-you-go endpoint, set
`model="qwen3.5-flash"` in the request or in the task defaults above. This
does not require a separate provider or a Token Plan subscription. Qwen Cloud
requires completed billing information and a payment method for pay-as-you-go
access. A `403 AccessDenied.Unpurchased` response means that billing setup is
incomplete for the account; changing the model ID does not resolve it. See
[Qwen Cloud billing setup](https://docs.qwencloud.com/resources/billing-overview)
and its [error reference](https://docs.qwencloud.com/api-reference/preparation/error-messages).

The provider supports non-streaming text generation, chat, and JSON output.
Allowed request options are `max_tokens`, `temperature`, and `top_p`. It sends
no tools or web-search requests. Structured output uses Qwen Cloud's JSON object
mode and validates a supplied JSON Schema locally. Model IDs are never inferred
from Ollama; pass an ID explicitly or configure a task default or routing
profile. Model listing reports configured defaults only, because the documented
Qwen Cloud API does not specify an account-scoped model inventory endpoint.
Explicit model IDs are checked by the service when called. Health status reports
configuration readiness without making a billable model request.

The API URL and request format follow the
[Qwen Cloud first-call guide](https://docs.qwencloud.com/developer-guides/getting-started/first-api-call)
and [structured-output guide](https://docs.qwencloud.com/developer-guides/text-generation/structured-output).
