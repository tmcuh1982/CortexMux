"""Mocked OpenAI client and provider tests."""

from __future__ import annotations

import json

import httpx
import pytest

from cortexmux.core.exceptions import InvalidRequestError
from cortexmux.core.types import MessageRole
from cortexmux.providers.openai import OpenAIClient, OpenAIProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import (
    ChatRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
)


@pytest.mark.asyncio
async def test_openai_client_post_preserves_dictionary_return_type() -> None:
    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"output_text": "OK"})
        ),
    )
    client = OpenAIClient(
        "https://api.openai.com/v1",
        api_key="secret",
        timeout=10,
        client=http_client,
    )

    result = await client.post("/responses", {"input": "hello"}, request_id="local-id")

    assert result == {"output_text": "OK"}
    await client.close()


@pytest.mark.asyncio
async def test_openai_discovers_accessible_models_and_normalizes_responses() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-5.6-luna", "object": "model"},
                        {"id": "gpt-5.6-terra", "object": "model"},
                    ]
                },
            )
        payload = json.loads(request.content)
        bodies.append(payload)
        output = '{"total":50,"moyenne":16.67,"tendance":"hausse"}' if payload.get("text") else "OK"
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": output}]}
                ],
                "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="secret",
            timeout=10,
            client=http_client,
        )
    )

    models = await provider.list_models()
    chat = await provider.execute(
        ChatRequest(
            provider="openai",
            model="gpt-5.6-luna",
            messages=[ChatMessage(role=MessageRole.USER, content="reply OK")],
            options={"reasoning_effort": "none", "max_output_tokens": 32},
        )
    )
    schema = {
        "type": "object",
        "properties": {"total": {"type": "number"}},
        "required": ["total"],
    }
    structured = await provider.execute(
        StructuredOutputRequest(
            provider="openai",
            model="gpt-5.6-luna",
            prompt="calculate",
            json_schema=schema,
            options={"verbosity": "low"},
        )
    )

    assert [item.name for item in models] == ["gpt-5.6-luna", "gpt-5.6-terra"]
    assert chat.content == "OK"
    assert chat.usage is not None and chat.usage.total_tokens == 25
    assert structured.parsed["total"] == 50
    assert bodies[0]["store"] is False
    assert bodies[0]["reasoning"] == {"effort": "none"}
    assert bodies[1]["text"] == {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "cortexmux_response",
            "schema": schema,
            "strict": True,
        },
    }

    await provider.close()


@pytest.mark.asyncio
async def test_openai_sends_system_instructions_separately_from_user_input() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        output = "{}" if len(bodies) == 2 else "OK"
        return httpx.Response(200, json={"output_text": output})

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="secret",
            timeout=10,
            client=http_client,
        )
    )

    await provider.execute(
        TextGenerationRequest(
            provider="openai",
            model="gpt-5.6-luna",
            prompt="user text",
            system="text instructions",
        )
    )
    await provider.execute(
        StructuredOutputRequest(
            provider="openai",
            model="gpt-5.6-luna",
            prompt="structured user text",
            system="structured instructions",
        )
    )

    assert bodies[0]["instructions"] == "text instructions"
    assert bodies[0]["input"] == "user text"
    assert bodies[1]["instructions"] == "structured instructions"
    assert bodies[1]["input"] == "structured user text"
    await provider.close()


@pytest.mark.asyncio
async def test_openai_validates_and_sends_standard_service_tier_before_network() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"output_text": "OK"})

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="secret",
            timeout=10,
            client=http_client,
        )
    )

    await provider.execute(
        TextGenerationRequest(
            provider="openai",
            model="gpt-5.6-luna",
            prompt="hello",
            options={"service_tier": "default"},
        )
    )
    assert bodies[0]["service_tier"] == "default"

    with pytest.raises(InvalidRequestError, match="Invalid OpenAI request options"):
        await provider.execute(
            TextGenerationRequest(
                provider="openai",
                model="gpt-5.6-luna",
                prompt="hello",
                options={"service_tier": "unknown"},
            )
        )
    with pytest.raises(InvalidRequestError, match="Invalid OpenAI request options"):
        await provider.execute(
            TextGenerationRequest(
                provider="openai",
                model="gpt-5.6-luna",
                prompt="hello",
                options={"unknown_option": True},
            )
        )
    assert len(bodies) == 1
    await provider.close()


@pytest.mark.asyncio
async def test_gpt_6_astra_defaults_to_low_and_rejects_unsupported_effort() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"output_text": "OK", "model": "gpt-6-astra"})

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="secret",
            timeout=10,
            client=http_client,
        )
    )

    await provider.execute(
        TextGenerationRequest(provider="openai", model="gpt-6-astra", prompt="hello")
    )
    await provider.execute(
        TextGenerationRequest(
            provider="openai",
            model="gpt-6-astra",
            prompt="hello",
            options={"reasoning_effort": "max"},
        )
    )
    assert [body["reasoning"] for body in bodies] == [
        {"effort": "low"},
        {"effort": "max"},
    ]

    with pytest.raises(InvalidRequestError, match="not supported by the selected model"):
        await provider.execute(
            TextGenerationRequest(
                provider="openai",
                model="gpt-6-astra",
                prompt="hello",
                options={"reasoning_effort": "none"},
            )
        )
    assert len(bodies) == 2
    await provider.close()


@pytest.mark.asyncio
async def test_openai_request_timeout_overrides_provider_timeout() -> None:
    timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"output_text": "OK"})

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="secret",
            timeout=10,
            client=http_client,
        )
    )

    await provider.execute(
        TextGenerationRequest(
            provider="openai",
            model="gpt-5.6-luna",
            prompt="hello",
            timeout=0.25,
        )
    )
    await provider.execute(
        TextGenerationRequest(provider="openai", model="gpt-5.6-luna", prompt="hello")
    )

    assert set(timeouts[0].values()) == {0.25}
    assert set(timeouts[1].values()) == {10.0}
    await provider.close()


@pytest.mark.asyncio
async def test_openai_retries_transient_errors_and_reports_serializable_metadata() -> None:
    statuses = iter([429, 500, 200])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = next(statuses)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "redacted"}})
        return httpx.Response(
            200,
            headers={"x-request-id": "req_provider_123"},
            json={
                "output_text": "OK",
                "model": "gpt-6-astra-2026-08-01",
                "service_tier": "default",
            },
        )

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="do-not-expose",
            timeout=10,
            client=http_client,
            max_retries=2,
            retry_base_delay_seconds=0,
        )
    )

    response = await provider.execute(
        TextGenerationRequest(
            provider="openai",
            model="gpt-6-astra",
            prompt="hello",
            options={"service_tier": "default", "reasoning_effort": "high"},
        )
    )

    assert calls == 3
    assert response.model == "gpt-6-astra-2026-08-01"
    assert response.raw_metadata is not None
    assert response.raw_metadata == {
        "provider": "openai",
        "model": "gpt-6-astra-2026-08-01",
        "requested_service_tier": "default",
        "returned_service_tier": "default",
        "reasoning_effort": "high",
        "attempts": 3,
        "provider_request_id": "req_provider_123",
        "duration_seconds": response.raw_metadata["duration_seconds"],
    }
    assert response.raw_metadata["duration_seconds"] >= 0
    response.model_dump_json()
    await provider.close()


@pytest.mark.asyncio
async def test_openai_does_not_retry_authentication_errors_or_expose_secrets() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "contains do-not-expose"}})

    http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAIProvider(
        OpenAIClient(
            "https://api.openai.com/v1",
            api_key="do-not-expose",
            timeout=10,
            client=http_client,
            max_retries=2,
            retry_base_delay_seconds=0,
        )
    )

    with pytest.raises(Exception) as error:
        await provider.execute(
            TextGenerationRequest(provider="openai", model="gpt-5.6-luna", prompt="hello")
        )
    assert calls == 1
    assert "do-not-expose" not in str(error.value)
    assert "do-not-expose" not in repr(getattr(error.value, "context", {}))
    await provider.close()
