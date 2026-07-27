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
