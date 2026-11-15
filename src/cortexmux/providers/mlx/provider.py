"""MLX text, chat and verified JSON responses."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cortexmux.behavior import BehaviorBinding
from cortexmux.core.behavior_runtime import execute_behavior, validate_bindings
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
from cortexmux.providers.mlx.client import MLXClient
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


class MLXOptions(BaseModel):
    """Whitelisted Chat Completions sampling controls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class MLXProvider(BaseProvider):
    """Route explicit model IDs to MLX without server-side tools."""

    name = "mlx"

    def __init__(
        self,
        client: MLXClient,
        *,
        models: list[str] | None = None,
        name: str = "mlx",
        server_version: str | None = None,
        check_openapi_version: bool = False,
        behavior_profiles: dict[str, BehaviorBinding] | None = None,
    ) -> None:
        self.name = name
        self.client = client
        self._server_version = server_version
        self._behaviors = dict(behavior_profiles or {})
        validate_bindings(self._behaviors, self.name)
        self._check_openapi_version = check_openapi_version
        self._models = tuple(dict.fromkeys(models or []))

    async def healthcheck(self) -> HealthStatus:
        """Report configuration readiness without contacting the server."""
        return HealthStatus(
            provider=self.name,
            available=True,
            message="MLX is configured; connectivity has not been verified.",
        )

    async def list_models(self) -> list[ModelInfo]:
        """List configured default model IDs without contacting the server."""
        return [ModelInfo(name=name, provider=self.name) for name in self._models]

    def accepts_unlisted_model(self, model: str) -> bool:
        """Defer acceptance of an explicit model ID to the MLX API."""
        return bool(model.strip())

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the implemented non-streaming text tasks."""
        binding = self._behaviors.get(model or "")
        tasks = frozenset(TaskType(t) for t in binding.profile.rules.tasks) if binding else _TASKS
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=tasks,
                structured_output=True,
                locally_hosted=True,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this adapter handles a task."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Validate and normalize a completed text or JSON answer."""
        binding = self._behaviors.get(request.model or "")
        if binding is not None:
            return await execute_behavior(
                request,
                binding,
                provider=self.name,
                server_version=self._server_version,
                client=self.client,
                observe_version=(
                    lambda: self.client.server_version(
                        request_id=request.request_id, timeout=request.timeout
                    )
                )
                if self._check_openapi_version
                else None,
            )
        context = {"provider": self.name, "request_id": request.request_id}
        if not isinstance(request, (TextGenerationRequest, ChatRequest, StructuredOutputRequest)):
            raise UnsupportedTaskError("MLX does not implement this task.", **context)
        if not request.model or not request.model.strip():
            raise InvalidRequestError("MLX requests require a model.", **context)
        if isinstance(request, (TextGenerationRequest, ChatRequest)) and request.stream:
            raise UnsupportedTaskError("MLX streaming is not implemented.", **context)
        if isinstance(request, StructuredOutputRequest) and request.think is not None:
            raise UnsupportedTaskError("MLX thinking controls are not implemented.", **context)
        try:
            options = MLXOptions.model_validate(request.options)
        except ValidationError:
            raise InvalidRequestError("Invalid MLX request options.", **context) from None

        messages: list[dict[str, str]] = []
        if isinstance(request, ChatRequest):
            for message in request.messages:
                if message.images or message.role == MessageRole.TOOL:
                    raise UnsupportedTaskError(
                        "MLX supports text-only chat without tools.", **context
                    )
                messages.append({"role": message.role.value, "content": message.content})
            if not any(message.role == MessageRole.USER for message in request.messages):
                raise InvalidRequestError("MLX chat requires a user message.", **context)
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
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Return only valid JSON."
                    + (
                        " Match this JSON Schema: " + json.dumps(request.json_schema)
                        if request.json_schema is not None
                        else ""
                    ),
                },
            )

        data = await self.client.complete(
            payload, request_id=request.request_id, timeout=request.timeout
        )
        content = _content(data, request.request_id, self.name)
        actual_model = data.get("model")
        metadata: dict[str, Any] = {}
        if isinstance(data.get("id"), str):
            metadata["provider_response_id"] = data["id"]
        common = {
            "provider": self.name,
            "model": actual_model if isinstance(actual_model, str) else request.model,
            "request_id": request.request_id,
            "content": content,
            "usage": _usage(data),
            "raw_metadata": metadata,
        }
        if isinstance(request, StructuredOutputRequest):
            try:
                parsed = json.loads(content, parse_constant=_reject_constant)
                if request.json_schema is not None:
                    validate_json_schema(parsed, request.json_schema)
            except ValueError:
                raise StructuredOutputValidationError(
                    "MLX returned invalid structured output.", **context
                ) from None
            return StructuredResponse(**common, parsed=parsed)
        if isinstance(request, ChatRequest):
            return ChatResponse(**common)
        return TextResponse(**common)

    async def close(self) -> None:
        """Close the instance-owned transport."""
        await self.client.close()


def _content(data: dict[str, Any], request_id: str, provider: str) -> str:
    context = {"provider": provider, "request_id": request_id}
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ProviderResponseError("MLX response has invalid choices.", **context)
    choice = choices[0]
    message = choice.get("message")
    if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
        raise ProviderResponseError("MLX response is incomplete.", **context)
    content = message.get("content")
    if message.get("role") != "assistant" or not isinstance(content, str) or not content:
        raise ProviderResponseError("MLX response has no assistant text.", **context)
    if message.get("tool_calls"):
        raise ProviderResponseError("MLX returned an unexpected tool call.", **context)
    return content


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite constants are not valid JSON")


def _usage(data: dict[str, Any]) -> UsageMetadata | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    def count(key: str) -> int | None:
        value = usage.get(key)
        return value if type(value) is int and value >= 0 else None

    return UsageMetadata(
        prompt_tokens=count("prompt_tokens"),
        completion_tokens=count("completion_tokens"),
        total_tokens=count("total_tokens"),
    )
