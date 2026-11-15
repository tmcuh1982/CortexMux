"""Offline contract tests for the Qwen Cloud adapter."""

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
from cortexmux.core.types import MessageRole
from cortexmux.facade import CortexMux
from cortexmux.providers.qwencloud import QwenCloudClient, QwenCloudProvider
from cortexmux.providers.qwencloud.client import validate_qwencloud_base_url
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest, TextGenerationRequest


def _provider(handler: httpx.MockTransport) -> QwenCloudProvider:
    client = httpx.AsyncClient(transport=handler)
    return QwenCloudProvider(
        QwenCloudClient(
            "https://maas.qwencloudapi.com/compatible-mode/v1", api_key="test-key", client=client
        ),
        models=["qwen3.8-flash", "qwen3.8-max"],
    )


def _answer(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "completion-1",
            "model": "qwen3.8-flash",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        },
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://maas.qwencloudapi.com/compatible-mode/v1",
        "https://user:secret@maas.qwencloudapi.com/compatible-mode/v1",
        "https://maas.qwencloudapi.com/compatible-mode/v1?token=secret",
    ],
)
def test_remote_url_rejects_insecure_or_sensitive_parts(url: str) -> None:
    with pytest.raises(ConfigurationError):
        validate_qwencloud_base_url(url)


@pytest.mark.asyncio
async def test_text_request_uses_qwen_endpoint_and_bearer_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url) == "https://maas.qwencloudapi.com/compatible-mode/v1/chat/completions"
        )
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen3.8-flash",
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
            "max_tokens": 64,
        }
        return _answer("Bonjour")

    provider = _provider(httpx.MockTransport(handler))
    response = await provider.execute(
        TextGenerationRequest(
            provider="qwencloud",
            model="qwen3.8-flash",
            prompt="Hello",
            system="Be brief",
            options={"max_tokens": 64},
        )
    )
    assert response.content == "Bonjour"
    assert response.usage is not None and response.usage.total_tokens == 11


@pytest.mark.asyncio
async def test_chat_preserves_roles_and_rejects_unknown_options() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Again"},
        ]
        return _answer("Again")

    provider = _provider(httpx.MockTransport(handler))
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ChatMessage(role=MessageRole.USER, content="Again"),
    ]
    assert (
        await provider.execute(ChatRequest(model="qwen3.8-flash", messages=messages))
    ).content == "Again"
    with pytest.raises(InvalidRequestError):
        await provider.execute(
            ChatRequest(model="qwen3.8-flash", messages=messages, options={"tools": "remote"})
        )


@pytest.mark.asyncio
async def test_structured_output_is_verified_locally() -> None:
    responses = iter([_answer('{"count":2}'), _answer('{"count":"wrong"}')])

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][0]["content"] == "Return only valid JSON."
        return next(responses)

    provider = _provider(httpx.MockTransport(handler))
    request = StructuredOutputRequest(
        model="qwen3.8-flash",
        prompt="Count items",
        json_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    )
    assert (await provider.execute(request)).parsed == {"count": 2}
    with pytest.raises(StructuredOutputValidationError):
        await provider.execute(request)


@pytest.mark.asyncio
async def test_incomplete_response_is_not_trusted() -> None:
    provider = _provider(
        httpx.MockTransport(lambda _request: _answer("partial", finish_reason="length"))
    )
    with pytest.raises(ProviderResponseError):
        await provider.execute(TextGenerationRequest(model="qwen3.8-flash", prompt="Hello"))


def test_opt_in_and_remote_approval_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "qwencloud": {
                    "enabled": True,
                    "api_key": "test-key",
                    "defaults": {"chat": "qwen3.8-flash"},
                },
            }
        }
    )
    with pytest.raises(RemoteHostNotAllowedError):
        CortexMux(config)
    config.core.approved_hosts.add("maas.qwencloudapi.com")
    monkeypatch.setattr("cortexmux.facade.QwenCloudClient", lambda *_args, **_kwargs: object())
    mux = CortexMux(config)
    provider = mux.registry.get("qwencloud")
    assert (
        config.model_for(
            ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")]).task,
            "qwencloud",
        )
        == "qwen3.8-flash"
    )
    assert provider.accepts_unlisted_model("qwen3.8-flash")
    selected, model, _reason = mux.router.select(
        ChatRequest(
            provider="qwencloud", messages=[ChatMessage(role=MessageRole.USER, content="hi")]
        )
    )
    assert selected is provider and model == "qwen3.8-flash"
