"""DeepSeek text, chat and locally verified JSON responses."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    InvalidRequestError,
    ProviderResponseError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.json_schema import validate_json_schema, validate_json_schema_definition
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.deepseek.client import DeepSeekClient
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


class DeepSeekOptions(BaseModel):
    """Whitelisted DeepSeek Chat Completions sampling controls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class DeepSeekProvider(BaseProvider):
    """Route explicit model IDs to DeepSeek while preserving cacheable prefixes."""

    name = "deepseek"

    def __init__(self, client: DeepSeekClient, *, models: list[str] | None = None) -> None:
        self.client = client
        self._models = tuple(dict.fromkeys(models or []))

    async def healthcheck(self) -> HealthStatus:
        """Report configured readiness without making a billable request."""
        return HealthStatus(
            provider=self.name,
            available=True,
            message="DeepSeek is configured; connectivity has not been verified.",
        )

    async def list_models(self) -> list[ModelInfo]:
        """List model IDs configured for routing."""
        return [ModelInfo(name=name, provider=self.name) for name in self._models]

    def accepts_unlisted_model(self, model: str) -> bool:
        """Defer acceptance of an explicit model ID to the DeepSeek API."""
        return bool(model.strip())

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the implemented non-streaming text tasks."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=_TASKS,
                structured_output=True,
                locally_hosted=False,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this adapter handles a task."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Validate a request and normalize a completed DeepSeek response."""
        context = {"provider": self.name, "request_id": request.request_id}
        if not isinstance(request, (TextGenerationRequest, ChatRequest, StructuredOutputRequest)):
            raise UnsupportedTaskError("DeepSeek does not implement this task.", **context)
        if not request.model or not request.model.strip():
            raise InvalidRequestError("DeepSeek requests require a model.", **context)
        if isinstance(request, (TextGenerationRequest, ChatRequest)) and request.stream:
            raise UnsupportedTaskError("DeepSeek streaming is not implemented.", **context)
        if isinstance(request, StructuredOutputRequest) and request.think is not None:
            raise UnsupportedTaskError("DeepSeek thinking controls are not implemented.", **context)
        try:
            options = DeepSeekOptions.model_validate(request.options)
        except ValidationError:
            raise InvalidRequestError("Invalid DeepSeek request options.", **context) from None

        messages: list[dict[str, str]] = []
        if isinstance(request, ChatRequest):
            for message in request.messages:
                if message.images or message.role == MessageRole.TOOL:
                    raise UnsupportedTaskError(
                        "DeepSeek supports text-only chat without tools.", **context
                    )
                messages.append({"role": message.role.value, "content": message.content})
            if not any(message.role == MessageRole.USER for message in request.messages):
                raise InvalidRequestError("DeepSeek chat requires a user message.", **context)
        else:
            if request.system is not None:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            **options.model_dump(exclude_none=True),
        }
        if isinstance(request, StructuredOutputRequest):
            if request.json_schema is not None:
                try:
                    validate_json_schema_definition(request.json_schema)
                except ValueError:
                    raise InvalidRequestError(
                        "Invalid or unsupported JSON Schema.", **context
                    ) from None
            messages.insert(0, {"role": "system", "content": "Return only valid JSON."})
            payload["response_format"] = {"type": "json_object"}

        data = await self.client.complete(
            payload, request_id=request.request_id, timeout=request.timeout
        )
        usage = _usage(data)
        content = _content(data, request.request_id, usage)
        actual_model = data.get("model")
        metadata: dict[str, Any] = {}
        if isinstance(data.get("id"), str):
            metadata["provider_response_id"] = data["id"]
        common = {
            "provider": self.name,
            "model": actual_model if isinstance(actual_model, str) else request.model,
            "request_id": request.request_id,
            "content": content,
            "usage": usage,
            "raw_metadata": metadata,
        }
        if isinstance(request, StructuredOutputRequest):
            try:
                parsed = json.loads(content, parse_constant=_reject_constant)
                if request.json_schema is not None:
                    validate_json_schema(parsed, request.json_schema)
            except ValueError:
                raise StructuredOutputValidationError(
                    "DeepSeek returned invalid structured output.",
                    **context,
                    usage=_usage_context(usage),
                ) from None
            return StructuredResponse(**common, parsed=parsed)
        if isinstance(request, ChatRequest):
            return ChatResponse(**common)
        return TextResponse(**common)

    async def close(self) -> None:
        """Close the instance-owned transport."""
        await self.client.close()


def _content(data: dict[str, Any], request_id: str, usage: UsageMetadata | None) -> str:
    context = {
        "provider": "deepseek",
        "request_id": request_id,
        "usage": _usage_context(usage),
    }
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ProviderResponseError("DeepSeek response has invalid choices.", **context)
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ProviderResponseError(
            "DeepSeek response is incomplete.",
            **context,
            finish_reason=choice.get("finish_reason"),
        )
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str) or not content:
        raise ProviderResponseError("DeepSeek response has no assistant text.", **context)
    if message.get("tool_calls"):
        raise ProviderResponseError("DeepSeek returned an unexpected tool call.", **context)
    return content


def _usage(data: dict[str, Any]) -> UsageMetadata | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = _non_negative_int(usage.get("prompt_tokens"))
    hit = _non_negative_int(usage.get("prompt_cache_hit_tokens"))
    miss = _non_negative_int(usage.get("prompt_cache_miss_tokens"))
    details = usage.get("prompt_tokens_details")
    if hit is None and isinstance(details, dict):
        hit = _non_negative_int(details.get("cached_tokens"))
    if miss is None and prompt is not None and hit is not None and hit <= prompt:
        miss = prompt - hit
    return UsageMetadata(
        prompt_tokens=prompt,
        completion_tokens=_non_negative_int(usage.get("completion_tokens")),
        total_tokens=_non_negative_int(usage.get("total_tokens")),
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
    )


def _usage_context(usage: UsageMetadata | None) -> dict[str, int] | None:
    if usage is None:
        return None
    values = usage.model_dump(exclude_none=True)
    return values or None


def _non_negative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
