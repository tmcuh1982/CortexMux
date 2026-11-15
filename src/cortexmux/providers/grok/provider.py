"""Grok text, chat and locally validated structured output through xAI."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from cortexmux.core.capabilities import ProviderCapability, TemperatureRange
from cortexmux.core.exceptions import (
    CortexMuxError,
    InvalidRequestError,
    ProviderResponseError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.json_schema import validate_json_schema, validate_json_schema_definition
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.grok.client import GrokClient
from cortexmux.providers.grok.schemas import GrokRequestOptions
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


class GrokProvider(BaseProvider):
    """Execute explicitly enabled text requests without server-side tools or storage."""

    name = "grok"

    def __init__(self, client: GrokClient) -> None:
        self.client = client

    async def healthcheck(self) -> HealthStatus:
        """Check authenticated model discovery without exposing response bodies."""
        try:
            await self.list_models()
            return HealthStatus(provider=self.name, available=True, message="Grok is available.")
        except CortexMuxError:
            return HealthStatus(
                provider=self.name,
                available=False,
                message="Grok is unavailable or authentication failed.",
            )

    async def list_models(self) -> list[ModelInfo]:
        """List accessible language model IDs and aliases advertised by xAI."""
        data = await self.client.list_models()
        items = data.get("models")
        if not isinstance(items, list):
            raise ProviderResponseError("Grok model list is invalid.", provider=self.name)
        models: dict[str, ModelInfo] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ProviderResponseError("Grok model entry is invalid.", provider=self.name)
            modalities = item.get("output_modalities", ["text"])
            if not isinstance(modalities, list):
                raise ProviderResponseError(
                    "Grok model modalities are invalid.", provider=self.name
                )
            if "text" not in modalities:
                continue
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) or not alias for alias in aliases
            ):
                raise ProviderResponseError("Grok model aliases are invalid.", provider=self.name)
            for name in [item["id"], *aliases]:
                models[name] = ModelInfo(name=name, provider=self.name)
        return list(models.values())

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the adapter's non-streaming text capabilities."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=_TASKS,
                structured_output=True,
                locally_hosted=False,
                temperature=TemperatureRange(minimum=0, maximum=2),
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this adapter implements the requested task."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Validate a request and normalize only completed assistant text."""
        if not isinstance(request, (TextGenerationRequest, ChatRequest, StructuredOutputRequest)):
            raise UnsupportedTaskError("Grok does not implement this task.", provider=self.name)
        if not request.model or not request.model.strip():
            raise InvalidRequestError("Grok requests require a model.", provider=self.name)
        result = await self.client.respond(
            _payload(request), request_id=request.request_id, timeout=request.timeout
        )
        data = result.data
        content = _output_text(data, request.request_id)
        model = data.get("model")
        metadata: dict[str, Any] = {
            "attempts": result.attempts,
            "duration_seconds": result.duration_seconds,
            "status": "completed",
        }
        if isinstance(data.get("id"), str):
            metadata["provider_response_id"] = data["id"]
        common = {
            "provider": self.name,
            "model": model if isinstance(model, str) else request.model,
            "request_id": request.request_id,
            "content": content,
            "usage": _usage(data),
            "raw_metadata": metadata,
        }
        if isinstance(request, StructuredOutputRequest):
            try:
                parsed = json.loads(content, parse_constant=_reject_json_constant)
                if request.json_schema is not None:
                    validate_json_schema(parsed, request.json_schema)
            except ValueError:
                raise StructuredOutputValidationError(
                    "Grok returned invalid structured output.",
                    provider=self.name,
                    request_id=request.request_id,
                ) from None
            return StructuredResponse(**common, parsed=parsed)
        if isinstance(request, ChatRequest):
            return ChatResponse(**common)
        return TextResponse(**common)

    async def close(self) -> None:
        """Release the Grok HTTP client."""
        await self.client.close()


def _payload(
    request: TextGenerationRequest | ChatRequest | StructuredOutputRequest,
) -> dict[str, Any]:
    context = {"provider": "grok", "request_id": request.request_id}
    try:
        options = GrokRequestOptions.model_validate(request.options)
    except ValidationError:
        raise InvalidRequestError("Invalid Grok request options.", **context) from None
    if isinstance(request, (TextGenerationRequest, ChatRequest)) and request.stream:
        raise UnsupportedTaskError("Grok streaming is not implemented.", **context)
    inputs: list[dict[str, str]] = []
    if isinstance(request, ChatRequest):
        for message in request.messages:
            if message.images or message.role == MessageRole.TOOL:
                raise UnsupportedTaskError("Grok supports text-only chat without tools.", **context)
            inputs.append({"role": message.role.value, "content": message.content})
        if not any(message.role == MessageRole.USER for message in request.messages):
            raise InvalidRequestError("Grok chat requires a user message.", **context)
    else:
        if request.system is not None:
            inputs.append({"role": "system", "content": request.system})
        inputs.append({"role": "user", "content": request.prompt})
    payload: dict[str, Any] = {
        "model": request.model,
        "input": inputs,
        "store": False,
        **options.model_dump(exclude_none=True),
    }
    if isinstance(request, StructuredOutputRequest):
        if request.think is not None:
            raise UnsupportedTaskError("Grok thinking controls are not implemented.", **context)
        output_format: dict[str, Any] = {"type": "json_object"}
        if request.json_schema is not None:
            try:
                validate_json_schema_definition(request.json_schema)
            except ValueError:
                raise InvalidRequestError(
                    "Invalid or unsupported JSON Schema.", **context
                ) from None
            output_format = {
                "type": "json_schema",
                "name": "cortexmux_response",
                "schema": request.json_schema,
                "strict": True,
            }
        else:
            inputs.insert(0, {"role": "system", "content": "Return only valid JSON."})
        payload["text"] = {"format": output_format}
    return payload


def _output_text(data: dict[str, Any], request_id: str) -> str:
    context = {"provider": "grok", "request_id": request_id}
    if data.get("status") != "completed" or data.get("error") or data.get("incomplete_details"):
        raise ProviderResponseError("Grok response failed or is incomplete.", **context)
    output = data.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("Grok response contains no output.", **context)
    fragments: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ProviderResponseError("Grok response contains invalid output.", **context)
        if item.get("type") == "reasoning":
            continue
        if (
            item.get("type") != "message"
            or item.get("role") != "assistant"
            or item.get("status", "completed") != "completed"
            or not isinstance(item.get("content"), list)
        ):
            raise ProviderResponseError("Grok response contains unexpected output.", **context)
        for part in item["content"]:
            if (
                not isinstance(part, dict)
                or part.get("type") != "output_text"
                or not isinstance(part.get("text"), str)
            ):
                raise ProviderResponseError("Grok refused or returned non-text output.", **context)
            fragments.append(part["text"])
    text = "".join(fragments)
    if not text:
        raise ProviderResponseError("Grok response contains no output text.", **context)
    return text


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite constants are not valid JSON")


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(data: dict[str, Any]) -> UsageMetadata | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    return UsageMetadata(
        prompt_tokens=_token_count(usage.get("input_tokens", usage.get("prompt_tokens"))),
        completion_tokens=_token_count(usage.get("output_tokens", usage.get("completion_tokens"))),
        total_tokens=_token_count(usage.get("total_tokens")),
    )
