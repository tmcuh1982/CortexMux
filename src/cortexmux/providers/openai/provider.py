"""OpenAI Responses API provider implementation."""

from __future__ import annotations

import json
from typing import Any

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    CortexMuxError,
    InvalidRequestError,
    ProviderResponseError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.json_schema import validate_json_schema
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.openai.client import OpenAIClient
from cortexmux.schemas.common import HealthStatus, ModelInfo, UsageMetadata
from cortexmux.schemas.requests import (
    ChatRequest,
    CortexRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
)
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    StructuredResponse,
    TextResponse,
)

_TASKS = frozenset({TaskType.TEXT_GENERATION, TaskType.CHAT, TaskType.STRUCTURED_OUTPUT})


class OpenAIProvider(BaseProvider):
    """Execute explicitly enabled remote text tasks through the Responses API."""

    name = "openai"

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    async def healthcheck(self) -> HealthStatus:
        """Check authenticated model discovery without exposing response data."""
        try:
            await self.client.get(OpenAIClient.MODELS)
            return HealthStatus(provider=self.name, available=True, message="OpenAI is available.")
        except CortexMuxError:
            return HealthStatus(
                provider=self.name,
                available=False,
                message="OpenAI is unavailable or authentication failed.",
            )

    async def list_models(self) -> list[ModelInfo]:
        """List only models actually accessible to the configured API account."""
        data = await self.client.get(OpenAIClient.MODELS)
        items = data.get("data")
        if not isinstance(items, list):
            raise ProviderResponseError("OpenAI model list is invalid.", provider=self.name)
        return [
            ModelInfo(name=item["id"], provider=self.name)
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the bounded Responses API features implemented here."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=_TASKS,
                structured_output=True,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this adapter implements the requested task."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Execute one normalized Responses API request."""
        if not request.model:
            raise InvalidRequestError("OpenAI requests require a model.", provider=self.name)
        if isinstance(request, TextGenerationRequest):
            data = await self._respond(request, request.prompt)
            return TextResponse(
                provider=self.name,
                model=request.model,
                request_id=request.request_id,
                content=_output_text(data, request.request_id),
                usage=_usage(data),
            )
        if isinstance(request, ChatRequest):
            inputs = [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ]
            data = await self._respond(request, inputs)
            return ChatResponse(
                provider=self.name,
                model=request.model,
                request_id=request.request_id,
                content=_output_text(data, request.request_id),
                usage=_usage(data),
            )
        if isinstance(request, StructuredOutputRequest):
            data = await self._respond(request, request.prompt, schema=request.json_schema)
            content = _output_text(data, request.request_id)
            try:
                parsed = json.loads(content)
                if request.json_schema is not None:
                    validate_json_schema(parsed, request.json_schema)
            except (json.JSONDecodeError, ValueError) as exc:
                raise StructuredOutputValidationError(
                    "OpenAI returned invalid structured output.",
                    provider=self.name,
                    request_id=request.request_id,
                ) from exc
            return StructuredResponse(
                provider=self.name,
                model=request.model,
                request_id=request.request_id,
                content=content,
                parsed=parsed,
                usage=_usage(data),
            )
        raise UnsupportedTaskError(
            "OpenAI does not implement this request type.",
            provider=self.name,
            task=request.task.value,
        )

    async def _respond(
        self,
        request: CortexRequest,
        inputs: str | list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": inputs,
            "store": False,
        }
        max_output = request.options.get("max_output_tokens")
        if isinstance(max_output, int):
            payload["max_output_tokens"] = max_output
        reasoning_effort = request.options.get("reasoning_effort")
        if isinstance(reasoning_effort, str):
            payload["reasoning"] = {"effort": reasoning_effort}
        text: dict[str, Any] = {}
        verbosity = request.options.get("verbosity")
        if isinstance(verbosity, str):
            text["verbosity"] = verbosity
        if schema is not None:
            text["format"] = {
                "type": "json_schema",
                "name": "cortexmux_response",
                "schema": schema,
                "strict": True,
            }
        if text:
            payload["text"] = text
        return await self.client.post(
            OpenAIClient.RESPONSES,
            payload,
            request_id=request.request_id,
        )

    async def close(self) -> None:
        """Close the OpenAI HTTP client."""
        await self.client.close()


def _output_text(data: dict[str, Any], request_id: str) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str):
        return direct
    output = data.get("output")
    if isinstance(output, list):
        fragments: list[str] = []
        for item in output:
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                continue
            for content in item["content"]:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
        if fragments:
            return "".join(fragments)
    raise ProviderResponseError(
        "OpenAI response contains no output text.",
        provider="openai",
        request_id=request_id,
    )


def _usage(data: dict[str, Any]) -> UsageMetadata | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")
    return UsageMetadata(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
        total_tokens=total if isinstance(total, int) else None,
    )
