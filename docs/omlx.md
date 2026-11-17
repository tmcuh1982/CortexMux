# oMLX structured output

CortexMux provides an opt-in `omlx` provider for calls with explicit model
bindings. The bundled profiles target oMLX 0.7.3 and these model IDs.

| Profile name | Model ID |
| --- | --- |
| `omlx-qwen-4bit` | `mlx-community/Qwen3.8-27B-4bit` |
| `omlx-gemma-4bit` | `gemma-4-31b-it-4bit` |

The profiles cover synchronous and asynchronous structured text with
`think=False` and a positive `max_tokens`. CortexMux passes
`chat_template_kwargs.enable_thinking=false` to oMLX, rejects responses that
report reasoning, and validates one JSON object against the requested schema.
The profiles accept a single JSON Markdown fence in profile normalization mode.

Bind the actual deployment and declare its version. The packaged manifests
include syntax fixtures and require explicit offline policy for this example.
Use a suitable live attestation if your application uses the live policy.
See [Behavior profiles](behavior-profiles.md) for the identity and qualification
rules. Configuration is opt-in and the ordinary endpoint approval rules apply.

```python
from cortexmux import CortexMux
from cortexmux.behavior import BehaviorBinding, bundled_profile
from cortexmux.core.config import CortexMuxConfig

profile = bundled_profile("omlx-gemma-4bit")
binding = BehaviorBinding(
    profile=profile,
    identity=profile.identity,
    qualification_policy="offline",
)
config = CortexMuxConfig.model_validate(
    {
        "providers": {
            "ollama": {"enabled": False},
            "comfyui": {"enabled": False},
            "omlx": {
                "enabled": True,
                "base_url": "http://127.0.0.1:8000/v1",
                "server_version": "0.7.3",
                "behavior_profiles": {profile.identity.model: binding},
            },
        },
    }
)
with CortexMux(config) as mux:
    reply = mux.structured(
        'Return {"status":"ok"}.',
        provider="omlx",
        model=profile.identity.model,
        behavior_profile=f"{profile.profile_id}@{profile.revision}",
        json_schema={"type": "object", "required": ["status"]},
        max_tokens=128,
        think=False,
    )
    print(reply.parsed)
```

Set `OMLX_API_KEY` for a deployment that requires credentials. The configuration
loader also supports `CORTEXMUX_OMLX_ENABLED` and `CORTEXMUX_OMLX_BASE_URL`.
