"""Offline Gemini contract, transport, and public facade regression tests."""

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
from cortexmux.providers.gemini import GeminiClient, GeminiProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import (
    ChatRequest,
    EmbeddingRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
)
from cortexmux.schemas.responses import ChatResponse, StructuredResponse

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-test"
SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}


def _response(text: str = "OK") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "modelVersion": "gemini-test-001",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 15,
        },
    }


def _model(name: str = MODEL) -> dict[str, Any]:
    return {"name": f"models/{name}", "supportedGenerationMethods": ["generateContent"]}


@pytest_asyncio.fixture
async def make_provider() -> AsyncIterator[Callable[..., GeminiProvider]]:
    clients: list[httpx.AsyncClient] = []

    def make(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> GeminiProvider:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
        clients.append(http)
        return GeminiProvider(
            GeminiClient(BASE_URL, api_key="test-secret", timeout=20, client=http, **kwargs)
        )

    yield make
    for client in clients:
        await client.aclose()


@pytest.mark.asyncio
async def test_text_contract_and_authentication(
    make_provider: Callable[..., GeminiProvider],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_URL}/models/{MODEL}:generateContent"
        assert request.headers["x-goog-api-key"] == "test-secret"
        assert request.extensions["timeout"]["read"] == 7
        assert json.loads(request.content) == {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "Be concise"}]},
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 100,
                "topP": 0.9,
                "topK": 10,
                "stopSequences": ["END"],
            },
        }
        data = _response()
        data["candidates"][0]["content"]["parts"] = [
            {"text": "private thought", "thought": True},
            {"text": "O"},
            {"text": "K"},
        ]
        return httpx.Response(200, json=data)

    provider = make_provider(handler)
    result = await provider.execute(
        TextGenerationRequest(
            prompt="Hello",
            system="Be concise",
            model=MODEL,
            timeout=7,
            options={
                "temperature": 0.5,
                "max_output_tokens": 100,
                "top_p": 0.9,
                "top_k": 10,
                "stop_sequences": ["END"],
            },
        )
    )
    assert result.content == "OK"
    assert result.model == "gemini-test-001"
    assert result.usage is not None
    assert result.usage.model_dump(exclude_none=True) == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 15,
    }
    assert result.raw_metadata["thoughts_tokens"] == 3
    assert result.raw_metadata["attempts"] == 1
    assert "test-secret" not in result.model_dump_json()
    assert "private thought" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_chat_role_mapping(make_provider: Callable[..., GeminiProvider]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["systemInstruction"] == {"parts": [{"text": "Rules"}]}
        assert payload["contents"] == [
            {"role": "user", "parts": [{"text": "First"}, {"text": "More"}]},
            {"role": "model", "parts": [{"text": "Answer"}]},
            {"role": "user", "parts": [{"text": "Next"}]},
        ]
        return httpx.Response(200, json=_response())

    result = await make_provider(handler).execute(
        ChatRequest(
            model=MODEL,
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="Rules"),
                ChatMessage(role=MessageRole.USER, content="First"),
                ChatMessage(role=MessageRole.USER, content="More"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Answer"),
                ChatMessage(role=MessageRole.USER, content="Next"),
            ],
        )
    )
    assert isinstance(result, ChatResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema", [None, SCHEMA])
async def test_structured_json_modes(
    make_provider: Callable[..., GeminiProvider],
    schema: dict[str, Any] | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        generation = json.loads(request.content)["generationConfig"]
        expected: dict[str, Any] = {"responseMimeType": "application/json"}
        if schema is not None:
            expected["responseJsonSchema"] = schema
        assert generation == expected
        return httpx.Response(200, json=_response('{"count": 3}'))

    result = await make_provider(handler).execute(
        StructuredOutputRequest(prompt="Count", model=MODEL, json_schema=schema)
    )
    assert isinstance(result, StructuredResponse)
    assert result.parsed == {"count": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ['{"count": "wrong"}', "not JSON", "{}"])
async def test_invalid_structured_output(
    make_provider: Callable[..., GeminiProvider],
    output: str,
) -> None:
    provider = make_provider(lambda _: httpx.Response(200, json=_response(output)))
    with pytest.raises(StructuredOutputValidationError):
        await provider.execute(
            StructuredOutputRequest(prompt="Count", model=MODEL, json_schema=SCHEMA)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_request",
    [
        TextGenerationRequest(prompt="hi"),
        TextGenerationRequest(prompt="hi", model="../models/other?key=secret"),
        TextGenerationRequest(prompt="hi", model=MODEL, options={"temperature": 3}),
        TextGenerationRequest(prompt="hi", model=MODEL, options={"temperature": float("nan")}),
        TextGenerationRequest(prompt="hi", model=MODEL, options={"tools": ["google_search"]}),
        TextGenerationRequest(prompt="hi", model=MODEL, options={"max_output_tokens": 0}),
        TextGenerationRequest(prompt="hi", model=MODEL, stream=True),
        EmbeddingRequest(inputs=["hi"], model=MODEL),
        StructuredOutputRequest(prompt="hi", model=MODEL, json_schema={"properties": []}),
        StructuredOutputRequest(prompt="hi", model=MODEL, think=True),
        ChatRequest(model=MODEL, messages=[ChatMessage(role=MessageRole.SYSTEM, content="rules")]),
        ChatRequest(model=MODEL, messages=[ChatMessage(role=MessageRole.TOOL, content="tool")]),
        ChatRequest(
            model=MODEL, messages=[ChatMessage(role=MessageRole.USER, content="hi", images=["abc"])]
        ),
        ChatRequest(model=MODEL, messages=[ChatMessage(role=MessageRole.ASSISTANT, content="hi")]),
        ChatRequest(
            model=MODEL,
            messages=[
                ChatMessage(role=MessageRole.USER, content="hi"),
                ChatMessage(role=MessageRole.SYSTEM, content="rules"),
            ],
        ),
    ],
)
async def test_invalid_requests_do_not_reach_generation(
    make_provider: Callable[..., GeminiProvider],
    invalid_request: Any,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid requests must not reach transport")

    with pytest.raises((InvalidRequestError, UnsupportedTaskError)):
        await make_provider(handler).execute(invalid_request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"candidates": []},
        {"candidates": [None]},
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {
            "candidates": [
                {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "partial"}]}}
            ]
        },
        {"candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "blocked"}]}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"functionCall": {}}]}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": ""}]}}]},
    ],
)
async def test_unusable_responses_are_errors(
    make_provider: Callable[..., GeminiProvider],
    data: dict[str, Any],
) -> None:
    provider = make_provider(lambda _: httpx.Response(200, json=data))
    with pytest.raises(ProviderResponseError):
        await provider.execute(TextGenerationRequest(prompt="hi", model=MODEL))


@pytest.mark.asyncio
async def test_catalog_pagination_and_capabilities(
    make_provider: Callable[..., GeminiProvider],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models"
        assert request.url.params["pageSize"] == "1000"
        if "pageToken" not in request.url.params:
            return httpx.Response(200, json={"models": [_model()], "nextPageToken": "next&?"})
        assert request.url.params["pageToken"] == "next&?"
        return httpx.Response(
            200,
            json={
                "models": [
                    _model("gemini-other"),
                    _model(),
                    {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
                ]
            },
        )

    provider = make_provider(handler)
    assert [m.name for m in await provider.list_models()] == [MODEL, "gemini-other"]
    assert (await provider.healthcheck()).available
    capability = (await provider.get_capabilities(MODEL))[0]
    assert capability.structured_output and not capability.locally_hosted
    assert not capability.streaming and not capability.image_input
    assert provider.supports(TaskType.CHAT) and not provider.supports(TaskType.VISION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {"models": "wrong"},
        {"models": [None]},
        {"models": [{"name": "models/../bad", "supportedGenerationMethods": ["generateContent"]}]},
        {"models": [], "nextPageToken": "repeated"},
        {"models": [], "nextPageToken": 123},
    ],
)
async def test_invalid_catalog(make_provider: Callable[..., GeminiProvider], data: Any) -> None:
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
        (429, 3, ProviderResponseError),
        (503, 3, ProviderResponseError),
        (302, 1, ProviderResponseError),
    ],
)
async def test_http_errors_and_redirects(
    make_provider: Callable[..., GeminiProvider],
    status: int,
    attempts: int,
    error: type[Exception],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.host == "generativelanguage.googleapis.com"
        return httpx.Response(
            status,
            json={"error": "sensitive body"},
            headers={"Location": "https://unapproved.example/secret"},
        )

    provider = make_provider(handler, retry_base_delay_seconds=0)
    with pytest.raises(error) as exc:
        await provider.execute(TextGenerationRequest(prompt="hi", model=MODEL))
    assert calls == attempts
    assert "sensitive body" not in str(exc.value)
    assert "test-secret" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,error",
    [
        (httpx.ReadTimeout, ProviderTimeoutError),
        (httpx.ConnectError, ProviderUnavailableError),
    ],
)
async def test_transport_errors(
    make_provider: Callable[..., GeminiProvider],
    failure: type[httpx.RequestError],
    error: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure("sensitive transport details", request=request)

    with pytest.raises(error) as exc:
        await make_provider(handler, max_retries=0).execute(
            TextGenerationRequest(prompt="hi", model=MODEL)
        )
    assert "sensitive" not in str(exc.value)
    assert exc.value.__suppress_context__


@pytest.mark.asyncio
async def test_retry_success_and_default_timeout(
    make_provider: Callable[..., GeminiProvider],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.extensions["timeout"]["read"] == 20
        return httpx.Response(429) if calls == 1 else httpx.Response(200, json=_response())

    result = await make_provider(handler, retry_base_delay_seconds=0).execute(
        TextGenerationRequest(prompt="hi", model=MODEL)
    )
    assert result.raw_metadata["attempts"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not JSON", b"[]"])
async def test_malformed_transport_json(
    make_provider: Callable[..., GeminiProvider],
    body: bytes,
) -> None:
    with pytest.raises(ProviderResponseError):
        await make_provider(lambda _: httpx.Response(200, content=body)).execute(
            TextGenerationRequest(prompt="hi", model=MODEL)
        )


@pytest.mark.asyncio
async def test_client_ownership() -> None:
    owned = GeminiClient(BASE_URL, api_key="test", timeout=10)
    await owned.close()
    assert owned._client.is_closed
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as http:
        borrowed = GeminiClient(BASE_URL, api_key="test", timeout=10, client=http)
        await borrowed.close()
        assert not http.is_closed


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@host/v1beta",
        f"{BASE_URL}?key=secret",
        f"{BASE_URL}#secret",
        "http://remote.example/v1beta",
        "relative/path",
    ],
)
def test_unsafe_endpoints(url: str) -> None:
    with pytest.raises(ConfigurationError) as exc:
        GeminiClient(url, api_key="test", timeout=10)
    assert "password" not in str(exc.value) and "secret" not in str(exc.value)


def _config(**gemini: Any) -> CortexMuxConfig:
    return CortexMuxConfig.load(
        environ={},
        overrides={
            "core": {"approved_hosts": ["generativelanguage.googleapis.com"]},
            "providers": {
                "ollama": {"enabled": False},
                "comfyui": {"enabled": False},
                "gemini": gemini,
            },
        },
    )


def test_configuration_and_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "present-but-not-opted-in")
    with CortexMux(config=_config()) as mux:
        assert "gemini" not in {p.name for p in mux.registry.list()}
    path = tmp_path / "config.toml"
    path.write_text('[providers.gemini]\nenabled = false\n[providers.gemini.defaults]\nchat = ""\n')
    config = CortexMuxConfig.load(
        path=path,
        environ={
            "CORTEXMUX_GEMINI_ENABLED": "true",
            "CORTEXMUX_GEMINI_BASE_URL": BASE_URL,
        },
        overrides={"providers": {"gemini": {"defaults": {"text_generation": MODEL}}}},
    )
    assert config.providers.gemini.enabled and config.providers.gemini.api_key is None
    assert config.model_for(TaskType.CHAT, "gemini") is None
    assert config.model_for(TaskType.TEXT_GENERATION, "gemini") == MODEL


def test_missing_key_and_unapproved_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
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
def test_facade_registration_routing_and_pydantic_output(
    monkeypatch: pytest.MonkeyPatch,
    explicit_key: bool,
) -> None:
    monkeypatch.setenv("CUSTOM_GEMINI_KEY", "environment-key")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        expected_key = "explicit-key" if explicit_key else "environment-key"
        assert request.headers["x-goog-api-key"] == expected_key
        if request.method == "GET":
            return httpx.Response(200, json={"models": [_model()]})
        return httpx.Response(200, json=_response('{"count": 3}'))

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "cortexmux.providers.gemini.client.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    config = _config(
        enabled=True,
        api_key="explicit-key" if explicit_key else None,
        api_key_env="CUSTOM_GEMINI_KEY",
        defaults={"text_generation": MODEL},
    )
    assert "explicit-key" not in repr(config)

    class Count(BaseModel):
        count: int

    with CortexMux(config=config) as mux:
        text = mux.generate("Count", provider="gemini")
        assert text.routing is not None and text.routing.selected_provider == "gemini"
        result = mux.structured("Count", provider="gemini", model=MODEL, response_model=Count)
        assert isinstance(result.parsed, Count) and result.parsed.count == 3
        with pytest.raises(ModelNotFoundError):
            mux.generate("No model", provider="gemini", model="missing")
        provider = mux.registry.get("gemini")
        assert isinstance(provider, GeminiProvider)
    assert provider.client._client.is_closed
    assert sum(request.method == "POST" for request in requests) == 2
