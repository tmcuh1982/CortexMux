# Configuration

Configuration precedence is Python overrides, environment, TOML, built-ins.
Without `CORTEXMUX_CONFIG`, CortexMux checks the `platformdirs` user
configuration directory for `config.toml`.

Supported environment variables include `CORTEXMUX_OLLAMA_BASE_URL`,
`CORTEXMUX_COMFYUI_BASE_URL`, `CORTEXMUX_OUTPUT_DIR`,
`CORTEXMUX_ALLOW_REMOTE_HOSTS`, `CORTEXMUX_LOG_LEVEL`, task model defaults, and
`CORTEXMUX_DEFAULT_COMFYUI_WORKFLOW`. Optional page retrieval uses
`CORTEXMUX_WEB_ENABLED`, `CORTEXMUX_WEB_ALLOWED_HOSTS`, and
`CORTEXMUX_WEB_ALLOW_PRIVATE_HOSTS`. Empty model strings mean unset.

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
chat = { provider = "ollama", model = "llama3.2:1b" }
structured_output = { provider = "ollama", model = "qwen3:4b" }

[routing.profiles.balanced]
chat = { provider = "ollama", model = "qwen3:4b" }
vision = { provider = "ollama", model = "gemma3:4b" }
embedding = { provider = "ollama", model = "nomic-embed-text:latest" }
```

With `validate_model_availability = true` (the default), CortexMux calls the
provider's model listing before execution and raises `ModelNotFoundError` if
the exact configured model is absent. This prevents an accidental request from
silently downloading or selecting another model. Set it to `false` only for a
provider whose model inventory cannot be listed reliably.

```python
with CortexMux.from_env(config_path="config.toml") as mux:
    quick = mux.chat("Résume ce texte", model_profile="fast")
    default = mux.chat("Explique cette décision")  # profile `balanced`
```

Provider headers are secret values. Do not commit credentials. Remote hosts
must be explicitly allowed and are never used as a silent fallback.

Mathematical verification is configured under `[data]` with
`max_calculation_claims`, `math_absolute_tolerance`, and
`math_relative_tolerance`. Tolerances must be non-negative decimals. Request
fields decide whether verification is enabled and whether every calculation
must pass.

Web retrieval is configured separately under `[web]`. It is disabled by
default and does not inherit `core.allow_remote_hosts`, which controls provider
endpoints. See [web extraction](web-extraction.md).
