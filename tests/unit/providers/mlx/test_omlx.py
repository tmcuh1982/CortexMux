"""Native oMLX registration and structured behavior through the facade."""

import json

import httpx
import pytest

from cortexmux import CortexMux
from cortexmux.behavior import BehaviorBinding, bundled_profile
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import CortexMuxError, ProviderResponseError, UnsupportedTaskError
from cortexmux.providers.mlx.client import MLXClient


def settings(profile, *, version=None):
    binding = BehaviorBinding(
        profile=profile, identity=profile.identity, qualification_policy="offline"
    )
    return CortexMuxConfig.model_validate(
        {
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "omlx": {
                    "enabled": True,
                    "server_version": version or profile.identity.server_version,
                    "behavior_profiles": {profile.identity.model: binding},
                },
            }
        }
    )


def mock_client(monkeypatch, http):
    def construct(*args, **kwargs):
        return MLXClient(*args, **kwargs, client=http)

    monkeypatch.setattr("cortexmux.facade.MLXClient", construct)


@pytest.mark.parametrize("name", ["omlx-qwen-4bit", "omlx-gemma-4bit"])
async def test_native_profile_uses_omlx_thinking_argument(name, monkeypatch):
    profile = bundled_profile(name)
    model = profile.identity.model
    schema = {"type": "object", "required": ["status"], "additionalProperties": True}

    def handler(request):
        assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
        wire = json.loads(request.content)
        assert wire == {
            "model": model,
            "stream": False,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Match this JSON Schema: "
                    + json.dumps(schema),
                },
                {"role": "user", "content": "Synthetic fixture"},
            ],
        }
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"status":"ok"}\n```',
                        },
                    }
                ],
                "usage": {
                    "completion_tokens": 11,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        mock_client(monkeypatch, http)
        async with CortexMux(settings(profile)) as mux:
            response = await mux.astructured(
                "Synthetic fixture",
                provider="omlx",
                model=model,
                think=False,
                max_tokens=128,
                json_schema=schema,
                behavior_profile=f"{profile.profile_id}@{profile.revision}",
            )
    assert response.provider == "omlx" and response.parsed == {"status": "ok"}
    assert response.raw_metadata["behavior"]["reasoning_tokens"] == 0
    assert response.raw_metadata["behavior"]["fingerprint"] == profile.fingerprint


@pytest.mark.parametrize("think", [True, None])
async def test_omlx_requires_an_explicit_reasoning_disabled_call(think, monkeypatch):
    profile = bundled_profile("omlx-gemma-4bit")

    def handler(request):
        pytest.fail("An out-of-profile call contacted the server")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        mock_client(monkeypatch, http)
        async with CortexMux(settings(profile)) as mux:
            with pytest.raises(UnsupportedTaskError):
                await mux.astructured(
                    "Synthetic fixture",
                    provider="omlx",
                    model=profile.identity.model,
                    think=think,
                    max_tokens=128,
                )


async def test_omlx_rejects_an_incompatible_configured_version(monkeypatch):
    profile = bundled_profile("omlx-qwen-4bit")

    def handler(request):
        pytest.fail("An incompatible version contacted the server")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        mock_client(monkeypatch, http)
        async with CortexMux(settings(profile, version="0.7.4")) as mux:
            with pytest.raises(CortexMuxError) as exc:
                await mux.astructured(
                    "Synthetic fixture",
                    provider="omlx",
                    model=profile.identity.model,
                    think=False,
                    max_tokens=128,
                )
    assert exc.value.context["behavior"]["category"] == "compatibility"


@pytest.mark.parametrize(
    ("extra_message", "tokens"),
    [({"reasoning_content": "synthetic"}, 0), ({}, 1)],
)
async def test_omlx_rejects_reported_reasoning(extra_message, tokens, monkeypatch):
    profile = bundled_profile("omlx-gemma-4bit")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": profile.identity.model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}", **extra_message},
                    }
                ],
                "usage": {"completion_tokens_details": {"reasoning_tokens": tokens}},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        mock_client(monkeypatch, http)
        async with CortexMux(settings(profile)) as mux:
            with pytest.raises(ProviderResponseError) as exc:
                await mux.astructured(
                    "Synthetic fixture",
                    provider="omlx",
                    model=profile.identity.model,
                    think=False,
                    max_tokens=128,
                )
    assert exc.value.context["behavior"]["category"] == "reasoning"
