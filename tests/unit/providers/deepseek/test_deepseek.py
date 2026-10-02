"""Offline contract tests for the DeepSeek provider."""

from __future__ import annotations

import json
import socket

import httpx
import pytest

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    ConfigurationError,
    InvalidRequestError,
    ProviderResponseError,
    RemoteHostNotAllowedError,
    StructuredOutputValidationError,
)
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.facade import CortexMux
from cortexmux.providers.deepseek import DeepSeekClient, DeepSeekProvider
from cortexmux.providers.deepseek.client import validate_deepseek_base_url
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest, TextGenerationRequest

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"


def _provider(handler: httpx.MockTransport) -> DeepSeekProvider:
    client = httpx.AsyncClient(transport=handler)
    return DeepSeekProvider(
        DeepSeekClient(BASE_URL, api_key="test-key", client=client), models=[MODEL]
    )


def _answer(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "completion-1",
            "model": MODEL,
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 3,
                "total_tokens": 83,
                "prompt_cache_hit_tokens": 64,
                "prompt_cache_miss_tokens": 16,
            },
        },
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com",
        "https://user:secret@api.deepseek.com",
        "https://api.deepseek.com?token=secret",
    ],
)
def test_remote_url_rejects_insecure_or_sensitive_parts(url: str) -> None:
    with pytest.raises(ConfigurationError):
        validate_deepseek_base_url(url)


@pytest.mark.asyncio
async def test_generation_preserves_cacheable_prefix_and_reports_cache_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Stable instructions"},
                {"role": "user", "content": "Stable document\n\nNew question"},
            ],
            "stream": False,
            "max_tokens": 64,
        }
        return _answer("OK")

    response = await _provider(httpx.MockTransport(handler)).execute(
        TextGenerationRequest(
            provider="deepseek",
            model=MODEL,
            prompt="Stable document\n\nNew question",
            system="Stable instructions",
            options={"max_tokens": 64},
        )
    )
    assert response.content == "OK"
    assert response.usage is not None
    assert response.usage.model_dump(exclude_none=True) == {
        "prompt_tokens": 80,
        "completion_tokens": 3,
        "total_tokens": 83,
        "cache_hit_tokens": 64,
        "cache_miss_tokens": 16,
    }


@pytest.mark.asyncio
async def test_chat_keeps_message_order_for_prefix_matching() -> None:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="stable"),
        ChatMessage(role=MessageRole.USER, content="document"),
        ChatMessage(role=MessageRole.ASSISTANT, content="answer"),
        ChatMessage(role=MessageRole.USER, content="follow-up"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["messages"] == [
            {"role": message.role.value, "content": message.content} for message in messages
        ]
        return _answer("OK")

    response = await _provider(httpx.MockTransport(handler)).execute(
        ChatRequest(model=MODEL, messages=messages)
    )
    assert response.content == "OK"


@pytest.mark.asyncio
async def test_cache_usage_supports_openai_compatible_details() -> None:
    response = _answer("OK")
    data = json.loads(response.content)
    data["usage"].pop("prompt_cache_hit_tokens")
    data["usage"].pop("prompt_cache_miss_tokens")
    data["usage"]["prompt_tokens_details"] = {"cached_tokens": 64}
    provider = _provider(httpx.MockTransport(lambda _: httpx.Response(200, json=data)))
    result = await provider.execute(TextGenerationRequest(model=MODEL, prompt="Hello"))
    assert result.usage is not None
    assert result.usage.cache_hit_tokens == 64
    assert result.usage.cache_miss_tokens == 16


@pytest.mark.asyncio
async def test_structured_output_is_verified_locally() -> None:
    responses = iter([_answer('{"count":2}'), _answer('{"count":"wrong"}')])

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0] == {
            "role": "system",
            "content": "Return only valid JSON.",
        }
        return next(responses)

    provider = _provider(httpx.MockTransport(handler))
    request = StructuredOutputRequest(
        model=MODEL,
        prompt="Count items as JSON",
        json_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    assert (await provider.execute(request)).parsed == {"count": 2}
    with pytest.raises(StructuredOutputValidationError) as caught:
        await provider.execute(request)
    assert caught.value.context["usage"] == {
        "prompt_tokens": 80,
        "completion_tokens": 3,
        "total_tokens": 83,
        "cache_hit_tokens": 64,
        "cache_miss_tokens": 16,
    }


@pytest.mark.asyncio
async def test_invalid_options_and_incomplete_output_are_rejected() -> None:
    provider = _provider(httpx.MockTransport(lambda _: _answer("partial", finish_reason="length")))
    with pytest.raises(InvalidRequestError):
        await provider.execute(
            TextGenerationRequest(model=MODEL, prompt="Hello", options={"tools": []})
        )
    with pytest.raises(ProviderResponseError) as caught:
        await provider.execute(TextGenerationRequest(model=MODEL, prompt="Hello"))
    assert caught.value.context == {
        "provider": "deepseek",
        "request_id": caught.value.context["request_id"],
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 3,
            "total_tokens": 83,
            "cache_hit_tokens": 64,
            "cache_miss_tokens": 16,
        },
    }
    assert "partial" not in json.dumps(caught.value.to_dict())


def test_opt_in_configuration_and_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))
        ],
    )
    config = CortexMuxConfig.model_validate(
        {
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "deepseek": {
                    "enabled": True,
                    "api_key": "test-key",
                    "defaults": {"chat": MODEL},
                },
            }
        }
    )
    with pytest.raises(RemoteHostNotAllowedError):
        CortexMux(config)
    config.core.approved_hosts.add("api.deepseek.com")
    monkeypatch.setattr("cortexmux.facade.DeepSeekClient", lambda *_args, **_kwargs: object())
    mux = CortexMux(config)
    provider = mux.registry.get("deepseek")
    assert config.model_for(TaskType.CHAT, "deepseek") == MODEL
    selected, model, _reason = mux.router.select(
        ChatRequest(
            provider="deepseek", messages=[ChatMessage(role=MessageRole.USER, content="hi")]
        )
    )
    assert selected is provider and model == MODEL


def test_environment_configuration() -> None:
    config = CortexMuxConfig.load(
        environ={
            "CORTEXMUX_DEEPSEEK_ENABLED": "true",
            "CORTEXMUX_DEEPSEEK_BASE_URL": BASE_URL,
        },
        overrides={"providers": {"deepseek": {"defaults": {"text_generation": MODEL}}}},
    )
    assert config.providers.deepseek.enabled
    assert config.providers.deepseek.base_url == BASE_URL
    assert config.model_for(TaskType.TEXT_GENERATION, "deepseek") == MODEL
