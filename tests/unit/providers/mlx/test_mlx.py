"""Offline contract tests for the MLX adapter."""

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
from cortexmux.providers.mlx import MLXClient, MLXProvider
from cortexmux.providers.mlx.client import validate_mlx_base_url
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest, TextGenerationRequest


def _provider(handler: httpx.MockTransport) -> MLXProvider:
    client = httpx.AsyncClient(transport=handler)
    return MLXProvider(
        MLXClient("http://192.168.1.50:8080/v1", api_key="test-key", client=client),
        models=["local-model", "other-model"],
    )


def _answer(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "completion-1",
            "model": "local-model",
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
        "ftp://192.168.1.50:8080/v1",
        "http://192.168.1.50:0/v1",
        "http://192.168.1.50:99999/v1",
        "http://192.168.1.50:bad/v1",
        "http://192.168.1.50:8080/wrong",
        "http://192.168.1.50:8080/v1#fragment",
        "https://user:secret@192.168.1.50/compatible-mode/v1",
        "http://192.168.1.50:8080/v1?token=secret",
    ],
)
def test_url_rejects_invalid_or_sensitive_parts(url: str) -> None:
    with pytest.raises(ConfigurationError):
        validate_mlx_base_url(url)


@pytest.mark.asyncio
async def test_text_request_uses_mlx_endpoint_and_bearer_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://192.168.1.50:8080/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "local-model",
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
            provider="mlx",
            model="local-model",
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
        await provider.execute(ChatRequest(model="local-model", messages=messages))
    ).content == "Again"
    with pytest.raises(InvalidRequestError):
        await provider.execute(
            ChatRequest(model="local-model", messages=messages, options={"tools": "remote"})
        )


@pytest.mark.asyncio
async def test_structured_output_is_verified_locally() -> None:
    responses = iter([_answer('{"count":2}'), _answer('{"count":"wrong"}')])

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "response_format" not in payload
        assert json.loads(
            payload["messages"][0]["content"].split("Match this JSON Schema: ")[1]
        ) == {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
        return next(responses)

    provider = _provider(httpx.MockTransport(handler))
    request = StructuredOutputRequest(
        model="local-model",
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
        await provider.execute(TextGenerationRequest(model="local-model", prompt="Hello"))


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
                "mlx": {
                    "enabled": True,
                    "base_url": "http://192.168.1.50:8080/v1",
                    "defaults": {"chat": "local-model"},
                },
            }
        }
    )
    with pytest.raises(RemoteHostNotAllowedError):
        CortexMux(config)
    config.core.approved_hosts.add("192.168.1.50")
    monkeypatch.setattr("cortexmux.facade.MLXClient", lambda *_args, **_kwargs: object())
    mux = CortexMux(config)
    provider = mux.registry.get("mlx")
    assert (
        config.model_for(
            ChatRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")]).task,
            "mlx",
        )
        == "local-model"
    )
    assert provider.accepts_unlisted_model("local-model")
    selected, model, _reason = mux.router.select(
        ChatRequest(provider="mlx", messages=[ChatMessage(role=MessageRole.USER, content="hi")])
    )
    assert selected is provider and model == "local-model"


def test_environment_and_disabled_default() -> None:
    assert not CortexMuxConfig().providers.mlx.enabled
    config = CortexMuxConfig.load(
        environ={
            "CORTEXMUX_MLX_ENABLED": "true",
            "CORTEXMUX_MLX_BASE_URL": "http://192.168.1.50:9090/v1",
        }
    )
    assert config.providers.mlx.enabled
    assert config.providers.mlx.base_url == "http://192.168.1.50:9090/v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 401, 404, 500])
async def test_http_failures_are_sanitized(status: int) -> None:
    from cortexmux.core.exceptions import CortexMuxError

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status, text="private server details", headers={"Location": "http://other/v1"}
            )
        )
    ) as transport:
        client = MLXClient("http://192.168.1.50:8080/v1", client=transport)
        with pytest.raises(CortexMuxError) as error:
            await client.complete({}, request_id="test")
        assert "private server details" not in str(error.value)
        assert not transport.is_closed
        await client.close()
        assert not transport.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host", "base_url"),
    [
        ("192.168.1.50", "http://192.168.1.50:9090/v1"),
        ("mac-studio.example-tailnet.ts.net", "https://mac-studio.example-tailnet.ts.net/v1"),
    ],
)
async def test_facade_sends_to_approved_host_without_key(
    monkeypatch: pytest.MonkeyPatch, host: str, base_url: str
) -> None:
    from cortexmux.providers.mlx import MLXClient as RealClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{base_url}/chat/completions"
        assert "authorization" not in request.headers
        assert json.loads(request.content)["model"] == "local-model"
        return _answer("Hello Studio")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        monkeypatch.delenv("MLX_API_KEY", raising=False)
        monkeypatch.setattr(
            "cortexmux.facade.MLXClient",
            lambda url, **kwargs: RealClient(url, client=transport, **kwargs),
        )
        config = CortexMuxConfig.model_validate(
            {
                "core": {"approved_hosts": [host]},
                "providers": {
                    "ollama": {"enabled": False},
                    "comfyui": {"enabled": False},
                    "mlx": {
                        "enabled": True,
                        "base_url": base_url,
                        "defaults": {"text_generation": "local-model"},
                    },
                },
            }
        )
        async with CortexMux(config) as mux:
            response = await mux.agenerate("Hello", provider="mlx")
            assert response.provider == "mlx"
            assert response.content == "Hello Studio"


@pytest.mark.asyncio
async def test_owned_client_is_closed() -> None:
    client = MLXClient("http://127.0.0.1:8080/v1")
    await client.close()
    assert client._client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "connection", "json", "array"])
async def test_transport_failures_are_typed(failure: str) -> None:
    from cortexmux.core.exceptions import ProviderTimeoutError, ProviderUnavailableError

    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("private detail", request=request)
        if failure == "connection":
            raise httpx.ConnectError("private detail", request=request)
        if failure == "json":
            return httpx.Response(200, text="private detail")
        return httpx.Response(200, json=[])

    expected = {
        "timeout": ProviderTimeoutError,
        "connection": ProviderUnavailableError,
        "json": ProviderResponseError,
        "array": ProviderResponseError,
    }[failure]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = MLXClient("http://127.0.0.1:8080/v1", client=transport)
        with pytest.raises(expected) as error:
            await client.complete({}, request_id="test", timeout=1)
        assert "private detail" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", ["stream", "images", "tools", "think"])
async def test_unsupported_features_never_send_requests(feature: str) -> None:
    from cortexmux.core.exceptions import UnsupportedTaskError

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unsupported features must fail before HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        provider = MLXProvider(MLXClient("http://127.0.0.1:8080/v1", client=transport))
        if feature == "think":
            request = StructuredOutputRequest(model="local-model", prompt="Hello", think=True)
        elif feature == "stream":
            request = TextGenerationRequest(model="local-model", prompt="Hello", stream=True)
        else:
            request = ChatRequest(
                model="local-model",
                messages=[
                    ChatMessage(
                        role=MessageRole.USER,
                        content="Hello",
                        images=["image"] if feature == "images" else [],
                    ),
                    *(
                        [ChatMessage(role=MessageRole.TOOL, content="tool")]
                        if feature == "tools"
                        else []
                    ),
                ],
            )
        with pytest.raises(UnsupportedTaskError):
            await provider.execute(request)


@pytest.mark.asyncio
async def test_lm_and_vlm_route_to_independent_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortexmux.providers.mlx import MLXClient as RealClient

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        payload = json.loads(request.content)
        assert payload["model"] == ("lm-model" if request.url.port == 8080 else "vlm-model")
        return _answer("ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        monkeypatch.setattr(
            "cortexmux.facade.MLXClient",
            lambda url, **kwargs: RealClient(url, client=transport, **kwargs),
        )
        config = CortexMuxConfig.model_validate(
            {
                "core": {"approved_hosts": ["studio.example"]},
                "providers": {
                    "ollama": {"enabled": False},
                    "comfyui": {"enabled": False},
                    "mlx_lm": {
                        "enabled": True,
                        "base_url": "http://studio.example:8080/v1",
                        "defaults": {"text_generation": "lm-model"},
                    },
                    "mlx_vlm": {
                        "enabled": True,
                        "base_url": "http://studio.example:8081/v1",
                        "defaults": {"text_generation": "vlm-model"},
                    },
                },
            }
        )
        async with CortexMux(config) as mux:
            for name in ("mlx_lm", "mlx_vlm"):
                response = await mux.agenerate("Hello", provider=name)
                assert response.provider == name
                assert response.routing.selected_provider == name
            assert [p.name for p in mux.registry.list() if p.name.startswith("mlx")] == [
                "mlx_lm",
                "mlx_vlm",
            ]
        assert seen == [
            "http://studio.example:8080/v1/chat/completions",
            "http://studio.example:8081/v1/chat/completions",
        ]


def test_named_mlx_environment_overrides() -> None:
    config = CortexMuxConfig.load(
        environ={
            "CORTEXMUX_MLX_LM_ENABLED": "true",
            "CORTEXMUX_MLX_LM_BASE_URL": "http://studio.example:8080/v1",
            "CORTEXMUX_MLX_VLM_ENABLED": "true",
            "CORTEXMUX_MLX_VLM_BASE_URL": "http://studio.example:8081/v1",
        }
    )
    assert config.providers.mlx_lm.enabled
    assert config.providers.mlx_vlm.enabled
    assert config.providers.mlx_lm.base_url != config.providers.mlx_vlm.base_url
    assert not config.providers.mlx.enabled
