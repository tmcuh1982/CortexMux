# Configuration

Configuration precedence is Python overrides, environment, TOML, built-ins.
Without `CORTEXMUX_CONFIG`, CortexMux checks the `platformdirs` user
configuration directory for `config.toml`.

Supported environment variables include `CORTEXMUX_OLLAMA_BASE_URL`,
`CORTEXMUX_OPENAI_ENABLED`, `CORTEXMUX_OPENAI_BASE_URL`,
`CORTEXMUX_GEMINI_ENABLED`, `CORTEXMUX_GEMINI_BASE_URL`,
`CORTEXMUX_GROK_ENABLED`, `CORTEXMUX_GROK_BASE_URL`,
`CORTEXMUX_DEEPSEEK_ENABLED`, `CORTEXMUX_DEEPSEEK_BASE_URL`,
`CORTEXMUX_QWENCLOUD_ENABLED`, `CORTEXMUX_QWENCLOUD_BASE_URL`,
`CORTEXMUX_TYPESAFE_ENABLED`, `CORTEXMUX_TYPESAFE_BASE_URL`,
`CORTEXMUX_COMFYUI_BASE_URL`, `CORTEXMUX_OUTPUT_DIR`,
`CORTEXMUX_ALLOW_REMOTE_HOSTS`, `CORTEXMUX_LOG_LEVEL`, task model defaults, and
`CORTEXMUX_DEFAULT_COMFYUI_WORKFLOW`. Optional page retrieval uses
`CORTEXMUX_WEB_ENABLED`, `CORTEXMUX_WEB_ALLOWED_HOSTS`, and
`CORTEXMUX_WEB_ALLOW_PRIVATE_HOSTS`. Empty model strings mean unset.

OpenAI remains disabled by default. When enabled, its API key is read from the
environment variable named by `providers.openai.api_key_env` (default
`OPENAI_API_KEY`), and `api.openai.com` must be explicitly approved under
`core.approved_hosts`. `providers.openai.max_retries` defaults to `2`, with
bounded full-jitter delays configured by `retry_base_delay_seconds` (default
`0.25`) and `retry_max_delay_seconds` (default `2`). See
[provider behavior](providers.md#openai-responses-api) and
[model qualification](model-qualification.md).

Gemini is also disabled by default. Enable `providers.gemini.enabled`, approve
`generativelanguage.googleapis.com` under `core.approved_hosts`, and set
`GEMINI_API_KEY` in the process environment. A key alone never enables Gemini.
`providers.gemini.api_key_env` can name another environment variable; an explicit
Python `api_key` (stored as `SecretStr`) takes precedence. The retry and timeout
settings have the same defaults as OpenAI above. Gemini requires HTTPS except
on loopback, rejects credentials/query strings/fragments in its base URL, and
never follows redirects. Use bare model IDs returned by `list_models("gemini")`
in `providers.gemini.defaults` or routing profiles. See the complete
[Gemini setup example](gemini.md).

Grok follows the same explicit activation policy: enable `providers.grok.enabled`,
approve `api.x.ai` under `core.approved_hosts`, and provide `XAI_API_KEY`.
`providers.grok.api_key_env` selects a different environment variable; an explicit
Python `api_key` takes precedence. Its base URL defaults to `https://api.x.ai/v1`,
and its timeout/retry settings match Gemini's. Configuring a key alone never
enables the provider or changes local task defaults. Use IDs or aliases returned
by `list_models("grok")`. See the [Grok setup guide](grok.md).

DeepSeek is disabled by default. Enable `providers.deepseek.enabled`, approve
`api.deepseek.com` under `core.approved_hosts`, and provide `DEEPSEEK_API_KEY`.
Its base URL defaults to `https://api.deepseek.com`. Configure explicit task
models under `providers.deepseek.defaults`. DeepSeek manages context caching
automatically; CortexMux preserves message order and reports cache-hit and
cache-miss token counts when the API returns them. See the
[DeepSeek setup and caching guide](deepseek.md).

Qwen Cloud is disabled by default. Enable `providers.qwencloud.enabled`, approve
`maas.qwencloudapi.com` under `core.approved_hosts`, and supply
`DASHSCOPE_API_KEY`. Configure a separate Qwen Cloud model for each task; local
Ollama routes remain independent. See the [Qwen Cloud setup guide](qwencloud.md).

TypeSafe decisions are also opt-in. Enable `providers.typesafe.enabled`, approve
`api.typesafe.ai` under `core.approved_hosts`, and supply `TYPESAFE_API_KEY`.
The provider only evaluates application-supplied typed questions; see the
[TypeSafe setup guide](typesafe.md).

## Project model profiles

Each project can commit a `config.toml` and choose exact, installed models by
task. `routing.active_profile` applies to requests that do not explicitly set a
provider or model. A request can select another configured profile through
`model_profile`; an explicit provider or model always wins. Model tags should
be pinned: `qwen3:4b` is unambiguous, whereas `qwen3` may resolve differently
as models are updated.

```toml
[routing]
active_profile = "balanced"
validate_model_availability = true

[routing.profiles.fast]
chat = { provider = "ollama", model = "qwen3:4b" }
structured_output = { provider = "ollama", model = "qwen3:4b" }

[routing.profiles.balanced]
chat = { provider = "ollama", model = "qwen3:4b" }
vision = { provider = "ollama", model = "ministral-3:8b" }
embedding = { provider = "ollama", model = "nomic-embed-text:latest" }

[routing.profiles.quality]
chat = { provider = "ollama", model = "qwen3.6:27b", options = { num_ctx = 8192, num_predict = 2048 } }
structured_output = { provider = "ollama", model = "qwen3.6:27b", options = { num_ctx = 8192, num_predict = 1024, temperature = 0 } }
vision = { provider = "ollama", model = "qwen3.6:27b", options = { num_ctx = 8192, num_predict = 1024 } }
```

Profile route options are merged into the request before execution. Explicit
request options take precedence. This is useful for large local models: the
official `qwen3.6:27b` model otherwise selects a context that can exceed the
practical unified-memory budget of a 24 GB Mac. The committed `quality`
example bounds it to 8192 tokens; it remains slower than the `balanced`
profile and is intended for deliberate high-quality requests.

With `validate_model_availability = true` (the default), CortexMux calls the
provider's model listing before execution and raises `ModelNotFoundError` if
the exact configured model is absent. This prevents an accidental request from
silently downloading or selecting another model. Set it to `false` only for a
provider whose model inventory cannot be listed reliably.

```python
with CortexMux.from_env(config_path="config.toml") as mux:
    quick = mux.chat("Résume ce texte", model_profile="fast")
    default = mux.chat("Explique cette décision")  # profile `balanced`
    detailed = mux.chat("Analyse cette architecture", model_profile="quality")
```

Provider headers are secret values. Do not commit credentials. Remote hosts
must be explicitly allowed and are never used as a silent fallback.

## Per-model temperature defaults

Providers may declare inclusive temperature bounds for each model through
their typed capabilities. `set_temperature()` validates those bounds before
installing an instance-scoped default for the exact provider/model pair:

```python
with CortexMux.from_env(config_path="config.toml") as mux:
    setting = mux.set_temperature(
        0.3,
        provider="ollama",
        model="qwen3.6:27b",
    )
    assert setting.minimum <= setting.value <= setting.maximum
    response = mux.chat("Analyse ce résultat", model_profile="quality")
```

The precedence is explicit request option, `set_temperature()` default,
profile option, then the provider/model default. Use `clear_temperature()` to
remove the instance default. Unsupported provider/model pairs and values
outside declared bounds raise `InvalidRequestError`; CortexMux never clamps a
temperature silently. The Ollama adapter deliberately enforces CortexMux's
normalized `0..2` range while custom providers can publish different ranges
for each model.

Mathematical verification is configured under `[data]` with
`max_calculation_claims`, `math_absolute_tolerance`, and
`math_relative_tolerance`. Tolerances must be non-negative decimals. Request
fields decide whether verification is enabled and whether every calculation
must pass.

Web retrieval is configured separately under `[web]`. It is disabled by
default and does not inherit `core.allow_remote_hosts`, which controls provider
endpoints. See [web extraction](web-extraction.md).

The optional `[mcp.capitalforge]` section starts a separate local process only
when explicitly enabled. It needs an absolute command, its fixed arguments,
and bounded request/shutdown timeouts. It does not inherit the parent process
environment. See [CapitalForge MCP](capitalforge-mcp.md) for the complete
configuration and read-only safety boundary.

## Codex subscription provider

The opt-in `codex` provider supports ChatGPT-managed login, text/structured
responses and account limits through local App Server stdio. See [Codex setup and limitations](codex.md).
