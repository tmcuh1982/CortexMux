"""Mocked OpenAI client and provider tests."""

from __future__ import annotations

import json

import httpx
import pytest

from cortexmux.core.types import MessageRole
from cortexmux.providers.openai import OpenAIClient, OpenAIProvider
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.requests import ChatRequest, StructuredOutputRequest


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
