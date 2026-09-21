"""Offline xAI transport, Grok request contracts and facade regression tests."""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel

from cortexmux import CortexMux
from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import (
    ConfigurationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RemoteHostNotAllowedError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.providers.grok import GrokClient, GrokProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import (
    ChatRequest,
    EmbeddingRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
)
from cortexmux.schemas.responses import ChatResponse, StructuredResponse

BASE_URL = "https://api.x.ai/v1"
MODEL = "grok-test"
SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}


def _response(text: str = "OK", **overrides: Any) -> dict[str, Any]:
    return {
        "id": "response-test",
        "status": "completed",
        "model": "grok-test-001",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        **overrides,
    }


@pytest_asyncio.fixture
async def make_provider() -> AsyncIterator[Callable[..., GrokProvider]]:
    clients: list[httpx.AsyncClient] = []

    def make(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> GrokProvider:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        clients.append(http)
        return GrokProvider(
            GrokClient(BASE_URL, api_key="test-secret", timeout=20, client=http, **kwargs)
        )

    yield make
    for client in clients:
        await client.aclose()


@pytest.mark.asyncio
async def test_generation_contract(make_provider: Callable[..., GrokProvider]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/responses"
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.extensions["timeout"]["read"] == 7
        assert json.loads(request.content) == {
            "model": MODEL,
            "input": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ],
            "store": False,
            "max_output_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
        }
        data = _response()
        data["output"].insert(0, {"type": "reasoning", "summary": [{"text": "private reasoning"}]})
        data["output"][1]["content"] = [
            {"type": "output_text", "text": "O"},
            {"type": "output_text", "text": "K"},
        ]
        return httpx.Response(200, json=data)

    result = await make_provider(handler).execute(
        TextGenerationRequest(
            prompt="Hi",
            system="Be concise",
            model=MODEL,
            timeout=7,
            options={"max_output_tokens": 100, "temperature": 0.5, "top_p": 0.9},
        )
    )
    assert result.content == "OK" and result.model == "grok-test-001"
    assert result.usage is not None
    assert result.usage.model_dump(exclude_none=True) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert result.raw_metadata["attempts"] == 1
    assert result.raw_metadata["provider_response_id"] == "response-test"
    assert "test-secret" not in result.model_dump_json()
    assert "private reasoning" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_chat_preserves_roles(make_provider: Callable[..., GrokProvider]) -> None:
    messages = [
        ChatMessage(role=role, content=role.value)
        for role in (
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"] == [
            {"role": message.role.value, "content": message.content} for message in messages
        ]
        return httpx.Response(200, json=_response())

    result = await make_provider(handler).execute(ChatRequest(model=MODEL, messages=messages))
    assert isinstance(result, ChatResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [None, SCHEMA])
async def test_json_output_contract(
    make_provider: Callable[..., GrokProvider],
    schema: dict[str, Any] | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["store"] is False
        if schema is None:
            assert payload["text"] == {"format": {"type": "json_object"}}
            assert "JSON" in payload["input"][0]["content"]
        else:
            assert payload["text"] == {
                "format": {
                    "type": "json_schema",
                    "name": "cortexmux_response",
                    "schema": schema,
                    "strict": True,
                }
            }
        return httpx.Response(200, json=_response('{"count": 3}'))

    result = await make_provider(handler).execute(
        StructuredOutputRequest(prompt="Count", model=MODEL, json_schema=schema)
    )
    assert isinstance(result, StructuredResponse) and result.parsed == {"count": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["not JSON", "{}", '{"count": "wrong"}', '{"count": NaN}'])
async def test_invalid_json_output(make_provider: Callable[..., GrokProvider], output: str) -> None:
    with pytest.raises(StructuredOutputValidationError):
        await make_provider(lambda _: httpx.Response(200, json=_response(output))).execute(
            StructuredOutputRequest(prompt="Count", model=MODEL, json_schema=SCHEMA)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_request",
    [
        TextGenerationRequest(prompt="Hi"),
        TextGenerationRequest(prompt="Hi", model="  "),
        TextGenerationRequest(prompt="Hi", model=MODEL, stream=True),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"tools": ["web_search"]}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"store": True}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"temperature": 3}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"temperature": float("nan")}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"top_p": -1}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"max_output_tokens": 0}),
        TextGenerationRequest(prompt="Hi", model=MODEL, options={"service_tier": "fast"}),
        StructuredOutputRequest(prompt="Hi", model=MODEL, json_schema={"properties": []}),
        StructuredOutputRequest(prompt="Hi", model=MODEL, think=True),
        EmbeddingRequest(inputs=["Hi"], model=MODEL),
        ChatRequest(model=MODEL, messages=[ChatMessage(role=MessageRole.TOOL, content="tool")]),
        ChatRequest(model=MODEL, messages=[ChatMessage(role=MessageRole.SYSTEM, content="system")]),
        ChatRequest(
            model=MODEL, messages=[ChatMessage(role=MessageRole.USER, content="Hi", images=["abc"])]
        ),
    ],
)
async def test_invalid_requests_do_not_generate(
    make_provider: Callable[..., GrokProvider],
    invalid_request: Any,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid request reached the API")

    with pytest.raises((InvalidRequestError, UnsupportedTaskError)):
        await make_provider(handler).execute(invalid_request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        _response(status="incomplete"),
        _response(status="failed"),
        _response(status="in_progress"),
        _response(error={"message": "sensitive details"}),
        _response(incomplete_details={"reason": "max_output_tokens"}),
        _response(output=[]),
        _response(output=[None]),
        _response(text=""),
        _response(output=[{"type": "function_call", "name": "unexpected"}]),
        _response(
            output=[{"type": "message", "role": "assistant", "status": "incomplete", "content": []}]
        ),
        _response(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "no"}],
                }
            ]
        ),
    ],
)
async def test_incomplete_refused_or_malformed_output(
    make_provider: Callable[..., GrokProvider],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ProviderResponseError) as exc:
        await make_provider(lambda _: httpx.Response(200, json=data)).execute(
            TextGenerationRequest(prompt="Hi", model=MODEL)
        )
    assert "sensitive details" not in str(exc.value)


@pytest.mark.asyncio
async def test_language_model_discovery(make_provider: Callable[..., GrokProvider]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/language-models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"id": MODEL, "aliases": ["grok-alias", MODEL], "output_modalities": ["text"]},
                    {"id": "image-model", "output_modalities": ["image"]},
                ]
            },
        )

    provider = make_provider(handler)
    assert [model.name for model in await provider.list_models()] == [MODEL, "grok-alias"]
    assert (await provider.healthcheck()).available
    capabilities = (await provider.get_capabilities(MODEL))[0]
    assert capabilities.structured_output and not capabilities.locally_hosted
    assert not capabilities.streaming and not capabilities.image_input
    assert provider.supports(TaskType.CHAT) and not provider.supports(TaskType.VISION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"models": "invalid"},
        {"models": [None]},
        {"models": [{"id": ""}]},
        {"models": [{"id": MODEL, "aliases": "wrong"}]},
        {"models": [{"id": MODEL, "aliases": [None]}]},
        {"models": [{"id": MODEL, "output_modalities": "text"}]},
    ],
)
async def test_invalid_model_catalog(make_provider: Callable[..., GrokProvider], data: Any) -> None:
    provider = make_provider(lambda _: httpx.Response(200, json=data))
    with pytest.raises(ProviderResponseError):
        await provider.list_models()
    assert not (await provider.healthcheck()).available


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,attempts,error",
    [
        (401, 1, ProviderResponseError),
        (403, 1, ProviderResponseError),
        (404, 1, ModelNotFoundError),
        (408, 3, ProviderResponseError),
        (429, 3, ProviderResponseError),
        (503, 3, ProviderResponseError),
        (302, 1, ProviderResponseError),
    ],
)
async def test_http_errors_and_redirects(
    make_provider: Callable[..., GrokProvider],
    status: int,
    attempts: int,
    error: type[Exception],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.host == "api.x.ai"
        return httpx.Response(
            status,
            json={"error": "sensitive body"},
            headers={"Location": "https://unapproved.example/secret"},
        )

    with pytest.raises(error) as exc:
        await make_provider(handler, retry_base_delay_seconds=0).execute(
            TextGenerationRequest(prompt="Hi", model=MODEL)
        )
    assert calls == attempts
    assert "sensitive body" not in str(exc.value) and "test-secret" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,error",
    [
        (httpx.ReadTimeout, ProviderTimeoutError),
        (httpx.ConnectError, ProviderUnavailableError),
    ],
)
async def test_transport_failures(
    make_provider: Callable[..., GrokProvider],
    failure: type[httpx.RequestError],
    error: type[Exception],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure("sensitive transport", request=request)

    with pytest.raises(error) as exc:
        await make_provider(handler, retry_base_delay_seconds=0).execute(
            TextGenerationRequest(prompt="Hi", model=MODEL)
        )
    assert calls == 3 and "sensitive transport" not in str(exc.value)
    assert exc.value.__suppress_context__


@pytest.mark.asyncio
async def test_retry_success(make_provider: Callable[..., GrokProvider]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.extensions["timeout"]["read"] == 20
        return httpx.Response(429) if calls == 1 else httpx.Response(200, json=_response())

    result = await make_provider(handler, retry_base_delay_seconds=0).execute(
        TextGenerationRequest(prompt="Hi", model=MODEL)
    )
    assert result.raw_metadata["attempts"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not JSON", b"[]"])
async def test_invalid_transport_json(
    make_provider: Callable[..., GrokProvider], body: bytes
) -> None:
    with pytest.raises(ProviderResponseError):
        await make_provider(lambda _: httpx.Response(200, content=body)).execute(
            TextGenerationRequest(prompt="Hi", model=MODEL)
        )


@pytest.mark.asyncio
async def test_owned_and_borrowed_clients() -> None:
    owned = GrokClient(BASE_URL, api_key="test", timeout=10)
    await owned.close()
    assert owned._client.is_closed
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as http:
        borrowed = GrokClient(BASE_URL, api_key="test", timeout=10, client=http)
        await borrowed.close()
        assert not http.is_closed


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@host/v1",
        f"{BASE_URL}?key=secret",
        f"{BASE_URL}#secret",
        "http://remote.example/v1",
        "relative/path",
    ],
)
def test_unsafe_endpoints(url: str) -> None:
    with pytest.raises(ConfigurationError) as exc:
        GrokClient(url, api_key="test", timeout=10)
    assert "password" not in str(exc.value) and "secret" not in str(exc.value)


def _config(**grok: Any) -> CortexMuxConfig:
    return CortexMuxConfig.load(
        environ={},
        overrides={
            "core": {"approved_hosts": ["api.x.ai"]},
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "grok": grok,
            },
        },
    )


def test_explicit_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "present-but-disabled")
    with CortexMux(config=_config()) as mux:
        assert "grok" not in {provider.name for provider in mux.registry.list()}
    path = tmp_path / "grok.toml"
    path.write_text('[providers.grok]\nenabled = false\n[providers.grok.defaults]\nchat = ""\n')
    config = CortexMuxConfig.load(
        path=path,
        environ={
            "CORTEXMUX_GROK_ENABLED": "true",
            "CORTEXMUX_GROK_BASE_URL": BASE_URL,
        },
        overrides={"providers": {"grok": {"defaults": {"text_generation": MODEL}}}},
    )
    assert config.providers.grok.enabled and config.providers.grok.api_key is None
    assert config.providers.grok.base_url == BASE_URL
    assert config.model_for(TaskType.CHAT, "grok") is None
    assert config.model_for(TaskType.TEXT_GENERATION, "grok") == MODEL


def test_missing_key_and_remote_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="API key"):
        CortexMux(config=_config(enabled=True))
    config = _config(enabled=True, api_key="test")
    config.core.approved_hosts.clear()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443)),
        ],
    )
    with pytest.raises(RemoteHostNotAllowedError):
        CortexMux(config=config)


@pytest.mark.parametrize("explicit_key", [False, True])
def test_facade_defaults_aliases_and_pydantic_output(
    monkeypatch: pytest.MonkeyPatch,
    explicit_key: bool,
) -> None:
    monkeypatch.setenv("CUSTOM_XAI_KEY", "environment-key")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        expected = "explicit-key" if explicit_key else "environment-key"
        assert request.headers["Authorization"] == f"Bearer {expected}"
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"id": MODEL, "aliases": ["grok-alias"]}]})
        payload = json.loads(request.content)
        assert payload["temperature"] == 0.4
        assert payload["store"] is False
        return httpx.Response(200, json=_response('{"count": 3}'))

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "cortexmux.providers.grok.client.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    config = _config(
        enabled=True,
        api_key="explicit-key" if explicit_key else None,
        api_key_env="CUSTOM_XAI_KEY",
        defaults={"text_generation": "grok-alias"},
    )
    assert "explicit-key" not in repr(config)

    class Count(BaseModel):
        count: int

    with CortexMux(config=config) as mux:
        mux.set_temperature(0.4, provider="grok", model="grok-alias")
        text = mux.generate("Count", provider="grok")
        assert text.routing is not None and text.routing.selected_provider == "grok"
        assert text.routing.selected_model == "grok-alias"
        result = mux.structured("Count", provider="grok", model="grok-alias", response_model=Count)
        assert isinstance(result.parsed, Count) and result.parsed.count == 3
        with pytest.raises(ModelNotFoundError):
            mux.generate("Hi", provider="grok", model="missing")
        provider = mux.registry.get("grok")
        assert isinstance(provider, GrokProvider)
    assert provider.client._client.is_closed
    assert sum(request.method == "POST" for request in requests) == 2
