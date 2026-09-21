"""Gemini text, chat and locally validated structured output."""

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
from cortexmux.providers.gemini.client import GeminiClient, validate_model_id
from cortexmux.providers.gemini.schemas import GeminiRequestOptions
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
_MAX_MODEL_PAGES = 100


class GeminiProvider(BaseProvider):
    """Execute explicitly enabled Google AI Studio requests through generateContent."""

    name = "gemini"

    def __init__(self, client: GeminiClient) -> None:
        self.client = client

    async def healthcheck(self) -> HealthStatus:
        """Check authenticated model discovery without disclosing API response bodies."""
        try:
            await self.list_models()
            return HealthStatus(provider=self.name, available=True, message="Gemini is available.")
        except CortexMuxError:
            return HealthStatus(
                provider=self.name,
                available=False,
                message="Gemini is unavailable or authentication failed.",
            )

    async def list_models(self) -> list[ModelInfo]:
        """List accessible generateContent model IDs across bounded catalog pages."""
        models: dict[str, ModelInfo] = {}
        page_token: str | None = None
        seen_tokens: set[str] = set()
        for _ in range(_MAX_MODEL_PAGES):
            data = await self.client.list_models(page_token)
            items = data.get("models", [])
            if not isinstance(items, list):
                raise ProviderResponseError("Gemini model list is invalid.", provider=self.name)
            for item in items:
                if not isinstance(item, dict):
                    raise ProviderResponseError(
                        "Gemini model entry is invalid.", provider=self.name
                    )
                methods = item.get("supportedGenerationMethods", [])
                if not isinstance(methods, list) or "generateContent" not in methods:
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.startswith("models/"):
                    raise ProviderResponseError("Gemini model name is invalid.", provider=self.name)
                try:
                    model_id = validate_model_id(name.removeprefix("models/"))
                except InvalidRequestError:
                    raise ProviderResponseError(
                        "Gemini model name is invalid.", provider=self.name
                    ) from None
                models[model_id] = ModelInfo(name=model_id, provider=self.name)
            next_token = data.get("nextPageToken")
            if next_token is None or next_token == "":
                return list(models.values())
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise ProviderResponseError(
                    "Gemini model pagination is invalid.", provider=self.name
                )
            seen_tokens.add(next_token)
            page_token = next_token
        raise ProviderResponseError("Gemini model pagination limit exceeded.", provider=self.name)

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe the text-only, non-streaming adapter capabilities."""
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
        """Return whether the adapter implements the requested task."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Validate a request, call Gemini, and normalize a complete text response."""
        if not isinstance(request, (TextGenerationRequest, ChatRequest, StructuredOutputRequest)):
            raise UnsupportedTaskError("Gemini does not implement this task.", provider=self.name)
        if not request.model:
            raise InvalidRequestError("Gemini requests require a model.", provider=self.name)
        validate_model_id(request.model)
        payload = _payload(request)
        result = await self.client.generate(
            request.model, payload, request_id=request.request_id, timeout=request.timeout
        )
        content = _output_text(result.data, request.request_id)
        actual_model = result.data.get("modelVersion")
        metadata: dict[str, Any] = {
            "attempts": result.attempts,
            "duration_seconds": result.duration_seconds,
            "finish_reason": "STOP",
        }
        raw_usage = result.data.get("usageMetadata")
        if isinstance(raw_usage, dict):
            thoughts = _token_count(raw_usage.get("thoughtsTokenCount"))
            if thoughts is not None:
                metadata["thoughts_tokens"] = thoughts
        common = {
            "provider": self.name,
            "model": actual_model if isinstance(actual_model, str) else request.model,
            "request_id": request.request_id,
            "content": content,
            "usage": _usage(result.data),
            "raw_metadata": metadata,
        }
        if isinstance(request, StructuredOutputRequest):
            try:
                parsed = json.loads(content)
                if request.json_schema is not None:
                    validate_json_schema(parsed, request.json_schema)
            except ValueError:
                raise StructuredOutputValidationError(
                    "Gemini returned invalid structured output.",
                    provider=self.name,
                    request_id=request.request_id,
                ) from None
            return StructuredResponse(**common, parsed=parsed)
        if isinstance(request, ChatRequest):
            return ChatResponse(**common)
        return TextResponse(**common)

    async def close(self) -> None:
        """Release the Gemini transport."""
        await self.client.close()


def _payload(
    request: TextGenerationRequest | ChatRequest | StructuredOutputRequest,
) -> dict[str, Any]:
    context = {"provider": "gemini", "request_id": request.request_id}
    try:
        options = GeminiRequestOptions.model_validate(request.options)
    except ValidationError:
        raise InvalidRequestError("Invalid Gemini request options.", **context) from None
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    if isinstance(request, (TextGenerationRequest, ChatRequest)) and request.stream:
        raise UnsupportedTaskError("Gemini streaming is not implemented.", **context)
    if isinstance(request, ChatRequest):
        for message in request.messages:
            if message.images or message.role == MessageRole.TOOL:
                raise UnsupportedTaskError(
                    "Gemini supports text-only chat without tools.", **context
                )
            if message.role == MessageRole.SYSTEM:
                if contents:
                    raise InvalidRequestError(
                        "Gemini system messages must precede chat turns.", **context
                    )
                system_parts.append({"text": message.content})
            else:
                role = "model" if message.role == MessageRole.ASSISTANT else "user"
                part = {"text": message.content}
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": role, "parts": [part]})
        if not contents or contents[0]["role"] != "user" or contents[-1]["role"] != "user":
            raise InvalidRequestError("Gemini chat must begin and end with a user turn.", **context)
    else:
        contents = [{"role": "user", "parts": [{"text": request.prompt}]}]
        if request.system is not None:
            system_parts.append({"text": request.system})
    generation: dict[str, Any] = {}
    for option, field in (
        ("max_output_tokens", "maxOutputTokens"),
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("top_k", "topK"),
        ("stop_sequences", "stopSequences"),
    ):
        value = getattr(options, option)
        if value is not None:
            generation[field] = value
    if isinstance(request, StructuredOutputRequest):
        if request.think is not None:
            raise UnsupportedTaskError("Gemini thinking controls are not implemented.", **context)
        generation["responseMimeType"] = "application/json"
        if request.json_schema is not None:
            try:
                validate_json_schema_definition(request.json_schema)
            except ValueError:
                raise InvalidRequestError(
                    "Invalid or unsupported JSON Schema.", **context
                ) from None
            generation["responseJsonSchema"] = request.json_schema
    payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def _output_text(data: dict[str, Any], request_id: str) -> str:
    context = {"provider": "gemini", "request_id": request_id}
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise ProviderResponseError("Gemini blocked the prompt.", **context)
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ProviderResponseError("Gemini response contains no candidate.", **context)
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise ProviderResponseError("Gemini response was blocked or incomplete.", **context)
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if isinstance(parts, list):
        fragments = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and not part.get("thought")
        ]
        if fragments and "".join(fragments):
            return "".join(fragments)
    raise ProviderResponseError("Gemini response contains no output text.", **context)


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _usage(data: dict[str, Any]) -> UsageMetadata | None:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    return UsageMetadata(
        prompt_tokens=_token_count(usage.get("promptTokenCount")),
        completion_tokens=_token_count(usage.get("candidatesTokenCount")),
        total_tokens=_token_count(usage.get("totalTokenCount")),
    )
