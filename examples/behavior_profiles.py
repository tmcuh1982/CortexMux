"""Two synthetic audited tasks through the public facade; no server required."""

import asyncio
import json
from typing import Any

import httpx

from cortexmux import CortexMux
from cortexmux.behavior import BehaviorBinding, bundled_profile
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import CortexMuxError
from cortexmux.providers.mlx.client import MLXClient
from cortexmux.providers.mlx.provider import MLXProvider


async def main() -> None:
    """Keep private audit in application-owned memory, never in public logs."""
    profile = bundled_profile("magistral")
    deployment = BehaviorBinding(
        profile=profile, identity=profile.identity, qualification_policy="offline"
    )
    private_records: list[dict[str, Any] | None] = []

    def synthetic_server(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '[THINK]Synthetic private text.[/THINK]{"answer":"ok"}',
                        },
                    }
                ],
            },
        )

    config = CortexMuxConfig.model_validate(
        {
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
            }
        }
    )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(synthetic_server)) as http,
        CortexMux(config) as mux,
    ):
        mux.register_provider(
            MLXProvider(
                MLXClient("http://127.0.0.1:8080", client=http),
                name="mlx_vlm",
                server_version="0.7.11",
                behavior_profiles={profile.identity.model: deployment},
            )
        )
        for schema, budget in [
            ({"type": "object", "required": ["answer"]}, 512),
            ({"type": "object", "required": ["relations"]}, 256),
        ]:
            try:
                result = await mux.astructured(
                    "Synthetic task",
                    provider="mlx_vlm",
                    model=profile.identity.model,
                    json_schema=schema,
                    max_tokens=budget,
                    capture_private=True,
                )
            except CortexMuxError as error:
                private_records.append(error.private_audit)
                print(error.context["behavior"]["category"])
            else:
                private_records.append(result.private_audit)
                print(result.raw_metadata["behavior"]["category"])
    assert len(private_records) == 2


if __name__ == "__main__":
    asyncio.run(main())
