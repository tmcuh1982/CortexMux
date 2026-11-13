# MLX server provider

The opt-in `mlx` provider connects directly to an MLX server on a Mac Studio
using non-streaming `/v1/chat/completions`. No Ollama process, MLX installation
on the client machine, or additional Python extra is required.

Save the following as `cortexmux.toml`, replacing the example IP, port, and
`your-served-model` with the Mac Studio's address and actual model identifier:

```toml
[core]
approved_hosts = ["192.168.1.50"]

[providers.mlx]
enabled = true
base_url = "http://192.168.1.50:8080/v1"
timeout_seconds = 120

[providers.mlx.defaults]
chat = "your-served-model"
text_generation = "your-served-model"
structured_output = "your-served-model"
```

```python
from cortexmux import CortexMux

with CortexMux.from_env(config_path="cortexmux.toml") as mux:
    response = mux.generate("Bonjour !", provider="mlx")
    print(response.content)
```

The server must listen on its LAN interface, rather than only on loopback,
and its port must be reachable from the client. HTTP is supported for an
explicitly approved LAN host; use HTTPS when the deployment provides it.
Approving the exact host avoids enabling remote access for all providers.
Redirects are never followed. URLs containing credentials, query strings,
fragments, invalid ports, or paths other than `/v1` are rejected.

## Tailscale DNS and HTTPS

`base_url` accepts DNS hostnames as well as IP addresses. For a Mac Studio
reachable through Tailscale, use the DNS hostname in both settings:

```toml
[core]
approved_hosts = ["mac-studio.example-tailnet.ts.net"]

[providers.mlx]
enabled = true
base_url = "https://mac-studio.example-tailnet.ts.net/v1"
```

Keep the task model defaults from the first example. HTTPS uses port 443 when
no port is specified; an explicit port can also be included. The machine running
CortexMux must be able to resolve and reach this Tailscale name. TLS certificate
verification stays enabled. A Tailscale HTTPS proxy can forward requests to an
MLX process listening on loopback on the Mac Studio.

To check the deployment separately from CortexMux:

```bash
curl -sS --max-time 15 https://mac-studio.example-tailnet.ts.net/v1/models
```

## Authentication and supported tasks

Environment overrides are `CORTEXMUX_MLX_ENABLED` and `CORTEXMUX_MLX_BASE_URL`.
You can also select the configuration file using `CORTEXMUX_CONFIG`.
Authentication is optional. If the deployment requires a bearer token, set
`MLX_API_KEY`, or use the secret `providers.mlx.api_key` configuration field.
The `api_key_env` setting can select another environment variable.

Supported tasks are text generation, text-only chat, and locally validated JSON
output. Allowed options are `max_tokens`, `temperature`, and `top_p`.
JSON output is prompted with the supplied schema and validated locally;
server-side constrained decoding is not assumed. Invalid JSON, schema
violations, truncated completions, and unexpected tool calls raise typed errors.
Streaming, tools, images, and thinking controls are not implemented.

Model listing returns configured task defaults only. Explicit model identifiers
are accepted and checked by the server when a request is sent. Health status
reports configuration readiness; it does not probe network connectivity.
The default routing still uses Ollama; select `provider="mlx"` explicitly or
set the relevant provider under `routing.defaults`.

The transport targets the Chat Completions format implemented by
[mlx-lm's server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py).
Unit tests use mocked HTTP responses; they do not require a running Mac Studio.

## Separate MLX-LM and MLX-VLM servers

Enable `mlx_lm` and `mlx_vlm` independently to use different ports on the same
host. Both use the text Chat Completions adapter; the `mlx_vlm` name does not
currently enable image inputs or the vision task. Its server must expose the
same text request/response format. The generic `mlx` configuration remains
supported independently.

The ports below are examples; use the ports actually exposed by your deployment.

```toml
[core]
approved_hosts = ["mac-studio.example-tailnet.ts.net"]

[providers.mlx_lm]
enabled = true
base_url = "http://mac-studio.example-tailnet.ts.net:8080/v1"

[providers.mlx_lm.defaults]
chat = "your-lm-model"
text_generation = "your-lm-model"

[providers.mlx_vlm]
enabled = true
base_url = "http://mac-studio.example-tailnet.ts.net:8081/v1"

[providers.mlx_vlm.defaults]
chat = "your-vlm-model"
text_generation = "your-vlm-model"
```

Select `provider="mlx_lm"` or `provider="mlx_vlm"` in calls or routing profiles.
For a TLS proxy, use `https://` and the proxy's externally exposed port for each
server. An HTTPS hostname on port 443 does not imply that backend ports 8080
and 8081 also support HTTPS.

Each named provider has its own timeout, defaults, and optional `api_key` or
`api_key_env`. Environment overrides are `CORTEXMUX_MLX_LM_ENABLED`,
`CORTEXMUX_MLX_LM_BASE_URL`, `CORTEXMUX_MLX_VLM_ENABLED`, and
`CORTEXMUX_MLX_VLM_BASE_URL`.

## Compatibility record: Magistral-Small-2509 MLX 8-bit

On 2026-11-13, the server's OpenAPI metadata reported **MLX-VLM 0.7.11**.
The campaign used non-streaming `POST /v1/chat/completions` and the CortexMux
evaluation profile `magistral-small-2509-mlx-8bit-r1`. This record describes
observed behavior with the evaluation adapter. The published adapter capabilities
are described above.

The canonical checkpoint was
[`lmstudio-community/Magistral-Small-2509-MLX-8bit`](https://huggingface.co/lmstudio-community/Magistral-Small-2509-MLX-8bit).
The deployment selected it using an exact server-side snapshot selector. The
server loaded Magistral on demand even when it was absent from the model list
after switching back to Qwen. The text control used
`mlx-community/Qwen3.8-27B-8bit` with standard structured-output validation.

| Probe | Observed result | Output tokens | Elapsed |
| --- | --- | ---: | ---: |
| Magistral, same fictional robot at 12 kg and 12,000 g | `stop`, direct JSON validated as `same_fact` | 50 | 2.850 s |
| Magistral, original synthetic blue PNG without a robot and gallery context | `stop`, direct JSON validated as `rejected` | 29 | 1.681 s |
| Magistral, explicit `max_tokens: 1` | HTTP 200 with `length`, rejected as truncated before JSON parsing | 1 | 0.322 s |
| Qwen text control, same mass comparison | JSON validated as `same_fact` via the standard path, with no compatibility profile selected | 148 | 14.907 s |
| Magistral, mass comparison with an explicit synthetic format example | HTTP 200 with `stop`, leading `[THINK]…[/THINK]` block and JSON validated as `same_fact` | 63 | 3.215 s |

The first format probe returned HTTP 200 and `stop` but failed structured
validation. The second supplied a synthetic example of the response format and
produced the leading block. CortexMux removed that prefix from the parsing copy
and applied the caller's schema to the remaining object. Nine checks confirmed
that the block, its text, and the private selector were absent from normalized
content and data, public metadata, string representations, and JSON, YAML and
Markdown exports. The intact server body remained available through the private
audit interface.

The text and image probes sent an explicit `max_tokens` and omitted
`response_format`. The PNG was a synthetic original. These are single-run
observations, including the elapsed times. The format example reproduced the
envelope on the real server. The ordinary mass and gallery probes returned
direct JSON, so callers should expect either form with this combination.
