"""Mocked Ollama client and provider tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cortexmux.core.exceptions import (
    InvalidRequestError,
    MCPToolNotAllowedError,
    ModelNotFoundError,
    ProviderResponseError,
    StructuredOutputValidationError,
    StructuredStreamInterruptedError,
)
from cortexmux.core.types import MessageRole
from cortexmux.mcp import MCPTool, MCPToolResult
from cortexmux.providers.ollama import OllamaClient, OllamaProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import (
    ChatRequest,
    EmbeddingRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
    VisionRequest,
)
from cortexmux.schemas.responses import StructuredStreamChunk, StructuredStreamCompleted


def provider_for(handler: httpx.MockTransport) -> OllamaProvider:
    """Create a provider backed by one mock transport."""
    http_client = httpx.AsyncClient(base_url="http://localhost:11434", transport=handler)
    return OllamaProvider(OllamaClient("http://localhost:11434", timeout=10, client=http_client))


class CapitalForgeMCPDouble:
    """In-memory read-only MCP double for native Ollama tool-loop tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def list_tools(self) -> list[MCPTool]:
        """Return only the five documented read tools."""
        return [
            MCPTool(
                name=name,
                description=f"Read {name}",
                input_schema={"type": "object", "properties": {}},
            )
            for name in (
                "capitalforge_source_catalog",
                "capitalforge_portfolio_summary",
                "capitalforge_positions",
                "capitalforge_zonebourse_signals",
                "capitalforge_public_signals",
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> MCPToolResult:
        """Record the server-bound arguments without receiving chat history."""
        self.calls.append((name, arguments))
        return MCPToolResult(
            tool_name=name,
            data={"source": name, "as_of": "2026-08-07", "notice": "Lecture seule."},
        )

    async def close(self) -> None:
        """Satisfy the provider cleanup contract."""


@pytest.mark.asyncio
async def test_capitalforge_tool_loop_uses_french_policy_and_bounded_read_context() -> None:
    """Give Ollama only safe tools, never a raw history or a mutation capability."""
    double = CapitalForgeMCPDouble()
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "capitalforge_source_catalog", "arguments": {}}},
                        {"function": {"name": "capitalforge_portfolio_summary", "arguments": {}}},
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Voici une synthèse locale."}},
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["role"] == "system"
        assert "French for French" in body["messages"][0]["content"]
        assert {item["function"]["name"] for item in body["tools"]} == {
            "capitalforge_source_catalog",
            "capitalforge_portfolio_summary",
            "capitalforge_positions",
            "capitalforge_zonebourse_signals",
            "capitalforge_public_signals",
        }
        assert "HISTORY_SECRET_MARKER" not in json.dumps(body.get("messages", [])[0])
        return httpx.Response(200, json=next(responses))

    http_client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    provider = OllamaProvider(
        OllamaClient("http://localhost:11434", timeout=10, client=http_client),
        capitalforge_mcp=double,  # type: ignore[arg-type]
    )
    response = await provider.chat_with_capitalforge(
        ChatRequest(
            provider="ollama",
            model="qwen3:4b",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content="Fais une analyse générale. HISTORY_SECRET_MARKER",
                )
            ],
        )
    )
    assert response.content == "Voici une synthèse locale."
    assert double.calls == [
        ("capitalforge_source_catalog", {}),
        ("capitalforge_portfolio_summary", {}),
    ]
    assert "HISTORY_SECRET_MARKER" not in repr(double.calls)
    await provider.close()


@pytest.mark.asyncio
async def test_capitalforge_tool_loop_rejects_more_than_eight_calls() -> None:
    """Refuse a model-driven tool loop before a ninth read reaches the server."""
    double = CapitalForgeMCPDouble()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "capitalforge_source_catalog", "arguments": {}}},
                        {"function": {"name": "capitalforge_portfolio_summary", "arguments": {}}},
                        {"function": {"name": "capitalforge_positions", "arguments": {"limit": 1}}},
                        {"function": {"name": "capitalforge_positions", "arguments": {"limit": 2}}},
                        {"function": {"name": "capitalforge_positions", "arguments": {"limit": 3}}},
                        {
                            "function": {
                                "name": "capitalforge_public_signals",
                                "arguments": {"limit": 1},
                            }
                        },
                        {
                            "function": {
                                "name": "capitalforge_public_signals",
                                "arguments": {"limit": 2},
                            }
                        },
                        {
                            "function": {
                                "name": "capitalforge_public_signals",
                                "arguments": {"limit": 3},
                            }
                        },
                        {
                            "function": {
                                "name": "capitalforge_zonebourse_signals",
                                "arguments": {"market": "usa"},
                            }
                        },
                    ],
                }
            },
        )

    http_client = httpx.AsyncClient(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    provider = OllamaProvider(
        OllamaClient("http://localhost:11434", timeout=10, client=http_client),
        capitalforge_mcp=double,  # type: ignore[arg-type]
    )
    with pytest.raises(MCPToolNotAllowedError, match="eight"):
        await provider.chat_with_capitalforge(
            ChatRequest(
                provider="ollama",
                model="qwen3:4b",
                messages=[ChatMessage(role=MessageRole.USER, content="Analyse.")],
            )
        )
    assert len(double.calls) == 8
    await provider.close()


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
async def test_structured_omits_unspecified_think_and_preserves_options() -> None:
    bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"response": '{"name":"ok"}'})

    provider = provider_for(httpx.MockTransport(handler))
    response = await provider.execute(
        StructuredOutputRequest(
            provider="ollama",
            model="m",
            prompt="hello",
            options={"temperature": 0.1},
        )
    )

    assert response.parsed == {"name": "ok"}
    assert bodies[0]["stream"] is False
    assert bodies[0]["format"] == "json"
    assert "think" not in bodies[0]
    assert bodies[0]["options"] == {"temperature": 0.1}
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["ministral-3:8b", "qwen3:4b"])
async def test_structured_stream_payload_fragments_and_validated_completion(
    model: str,
) -> None:
    bodies: list[dict[str, object]] = []
    stream = (
        b'{"response":"{\\"name\\":","done":false}\n'
        b'{"response":"\\"ok\\"","done":false}\n'
        b'{"response":"}","done":true,"prompt_eval_count":2,"eval_count":3}\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=stream)

    provider = provider_for(httpx.MockTransport(handler))
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    events = [
        event
        async for event in provider.stream(
            StructuredOutputRequest(
                provider="ollama",
                model=model,
                prompt="Return a name.",
                json_schema=schema,
                think=False,
            )
        )
    ]

    assert bodies == [
        {
            "model": model,
            "prompt": "Return a name.",
            "stream": True,
            "format": schema,
            "think": False,
            "options": {},
        }
    ]
    chunks = [event for event in events if isinstance(event, StructuredStreamChunk)]
    assert [chunk.content for chunk in chunks] == ['{"name":', '"ok"', "}"]
    completed = events[-1]
    assert isinstance(completed, StructuredStreamCompleted)
    assert completed.content == '{"name":"ok"}'
    assert completed.parsed == {"name": "ok"}
    assert completed.provider == "ollama"
    assert completed.model == model
    assert completed.request_id
    assert completed.usage is not None
    assert completed.usage.total_tokens == 5
    await provider.close()


@pytest.mark.asyncio
async def test_structured_stream_rejects_invalid_final_json_with_safe_excerpt() -> None:
    stream = b'{"response":"{\\"wrong\\":true}","done":true}\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream)

    provider = provider_for(httpx.MockTransport(handler))
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    seen: list[object] = []
    with pytest.raises(StructuredOutputValidationError) as error:
        async for event in provider.stream(
            StructuredOutputRequest(
                provider="ollama",
                model="ministral-3:8b",
                prompt="Return a name.",
                json_schema=schema,
            )
        ):
            seen.append(event)

    assert len(seen) == 1
    assert isinstance(seen[0], StructuredStreamChunk)
    assert len(str(error.value.context["safe_excerpt"])) <= 200
    await provider.close()


@pytest.mark.asyncio
async def test_structured_stream_requires_ollama_completion_marker() -> None:
    stream = b'{"response":"{\\"name\\":","done":false}\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream)

    provider = provider_for(httpx.MockTransport(handler))
    with pytest.raises(StructuredStreamInterruptedError):
        async for _ in provider.stream(
            StructuredOutputRequest(
                provider="ollama",
                model="qwen3:4b",
                prompt="Return a name.",
                think=False,
            )
        ):
            pass
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
    assert capabilities[0].temperature is not None
    assert capabilities[0].temperature.minimum == 0
    assert capabilities[0].temperature.maximum == 2
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
