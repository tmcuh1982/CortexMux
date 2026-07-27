"""Mocked Ollama client and provider tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cortexmux.core.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    ProviderResponseError,
    StructuredOutputValidationError,
)
from cortexmux.core.types import MessageRole
from cortexmux.providers.ollama import OllamaClient, OllamaProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import (
    ChatRequest,
    EmbeddingRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
    VisionRequest,
)


def provider_for(handler: httpx.MockTransport) -> OllamaProvider:
    """Create a provider backed by one mock transport."""
    http_client = httpx.AsyncClient(base_url="http://localhost:11434", transport=handler)
    return OllamaProvider(OllamaClient("http://localhost:11434", timeout=10, client=http_client))


@pytest.mark.asyncio
async def test_health_models_generation_chat_and_embeddings() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "1.2.3"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "m",
                            "size": 12,
                            "details": {"family": "test"},
                            "modified_at": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        if request.url.path == "/api/generate":
            assert body["stream"] is False
            return httpx.Response(
                200,
                json={"response": "generated", "prompt_eval_count": 2, "eval_count": 3},
            )
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "chat"}})
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 2], [3, 4]]})
        raise AssertionError(request.url.path)

    provider = provider_for(httpx.MockTransport(handler))
    assert (await provider.healthcheck()).version == "1.2.3"
    assert (await provider.list_models())[0].family == "test"
    generated = await provider.execute(
        TextGenerationRequest(provider="ollama", model="m", prompt="hello")
    )
    assert generated.content == "generated"
    assert generated.usage is not None and generated.usage.total_tokens == 5
    chatted = await provider.execute(
        ChatRequest(
            provider="ollama",
            model="m",
            messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        )
    )
    assert chatted.content == "chat"
    embedded = await provider.execute(
        EmbeddingRequest(provider="ollama", model="m", inputs=["a", "b"])
    )
    assert embedded.embeddings == [[1.0, 2.0], [3.0, 4.0]]
    await provider.close()


@pytest.mark.asyncio
async def test_structured_success_and_failure() -> None:
    responses = iter(
        [
            httpx.Response(200, json={"response": '{"name":"ok"}'}),
            httpx.Response(200, json={"response": '{"wrong":true}'}),
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    provider = provider_for(httpx.MockTransport(handler))
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    response = await provider.execute(
        StructuredOutputRequest(provider="ollama", model="m", prompt="hello", json_schema=schema)
    )
    assert response.parsed == {"name": "ok"}
    with pytest.raises(StructuredOutputValidationError):
        await provider.execute(
            StructuredOutputRequest(
                provider="ollama", model="m", prompt="hello", json_schema=schema
            )
        )
    await provider.close()


@pytest.mark.asyncio
async def test_vision_encodes_bytes_and_path(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"abc")
    seen: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "seen"}})

    provider = provider_for(httpx.MockTransport(handler))
    response = await provider.execute(
        VisionRequest(
            provider="ollama",
            model="vision",
            prompt="look",
            images=[image, b"xyz"],
        )
    )
    assert response.content == "seen"
    images = seen[0]["messages"][0]["images"]  # type: ignore[index]
    assert images == ["YWJj", "eHl6"]
    await provider.close()


@pytest.mark.asyncio
async def test_incremental_stream_and_malformed_data() -> None:
    good = b'{"response":"a","done":false}\n{"response":"b","done":true}\n'

    async def good_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=good)

    provider = provider_for(httpx.MockTransport(good_handler))
    chunks = [
        chunk.content
        async for chunk in provider.stream(
            TextGenerationRequest(model="m", prompt="hello", stream=True)
        )
    ]
    assert chunks == ["a", "b"]
    await provider.close()

    async def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json\n")

    bad_provider = provider_for(httpx.MockTransport(bad_handler))
    with pytest.raises(ProviderResponseError):
        async for _ in bad_provider.stream(
            TextGenerationRequest(model="m", prompt="hello", stream=True)
        ):
            pass
    await bad_provider.close()


@pytest.mark.asyncio
async def test_http_error_mapping() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    provider = provider_for(httpx.MockTransport(handler))
    with pytest.raises(ModelNotFoundError):
        await provider.list_models()
    assert not (await provider.healthcheck()).available
    await provider.close()


@pytest.mark.asyncio
async def test_capabilities_options_and_invalid_vision(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"})

    provider = provider_for(httpx.MockTransport(handler))
    capabilities = await provider.get_capabilities("m")
    assert capabilities[0].structured_output
    assert provider.supports(TextGenerationRequest(prompt="x").task)
    response = await provider.execute(
        TextGenerationRequest(
            model="m",
            prompt="hello",
            system="system",
            keep_alive="5m",
            options={"temperature": 0.1},
        )
    )
    assert response.content == "ok"
    assert bodies[0]["system"] == "system"
    assert bodies[0]["keep_alive"] == "5m"
    with pytest.raises(InvalidRequestError):
        await provider.execute(
            VisionRequest(model="m", prompt="look", images=["not base64 or a path"])
        )
    missing = tmp_path / "missing.png"
    with pytest.raises(InvalidRequestError):
        await provider.execute(VisionRequest(model="m", prompt="look", images=[missing]))
    await provider.close()
