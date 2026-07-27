# Configuration

Configuration precedence is Python overrides, environment, TOML, built-ins.
Without `CORTEXMUX_CONFIG`, CortexMux checks the `platformdirs` user
configuration directory for `config.toml`.

Supported environment variables include `CORTEXMUX_OLLAMA_BASE_URL`,
`CORTEXMUX_COMFYUI_BASE_URL`, `CORTEXMUX_OUTPUT_DIR`,
`CORTEXMUX_ALLOW_REMOTE_HOSTS`, `CORTEXMUX_LOG_LEVEL`, task model defaults, and
`CORTEXMUX_DEFAULT_COMFYUI_WORKFLOW`. Empty model strings mean unset.

Provider headers are secret values. Do not commit credentials. Remote hosts
must be explicitly allowed and are never used as a silent fallback.

Mathematical verification is configured under `[data]` with
`max_calculation_claims`, `math_absolute_tolerance`, and
`math_relative_tolerance`. Tolerances must be non-negative decimals. Request
fields decide whether verification is enabled and whether every calculation
must pass.
