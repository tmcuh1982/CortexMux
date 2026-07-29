"""Ollama provider implementation."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.exceptions import (
    CortexMuxError,
    InvalidRequestError,
    ProviderResponseError,
    StructuredOutputValidationError,
    UnsupportedTaskError,
)
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.ollama.client import OllamaClient
from cortexmux.providers.ollama.schemas import usage_from_ollama
from cortexmux.schemas.common import HealthStatus, ModelInfo
from cortexmux.schemas.requests import (
    ChatRequest,
    CortexRequest,
    EmbeddingRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
    VisionRequest,
)
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    EmbeddingResponse,
    StreamChunk,
    StructuredResponse,
    TextResponse,
    VisionResponse,
)

_TASKS = frozenset(
    {
        TaskType.TEXT_GENERATION,
        TaskType.CHAT,
        TaskType.STRUCTURED_OUTPUT,
        TaskType.VISION,
        TaskType.EMBEDDING,
    }
)


class OllamaProvider(BaseProvider):
    """Execute text, chat, vision, structured, and embedding tasks in Ollama."""

    name = "ollama"

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    async def healthcheck(self) -> HealthStatus:
        """Check Ollama's version endpoint and convert downtime to a status."""
        try:
            data = await self.client.get(OllamaClient.VERSION)
            return HealthStatus(
                provider=self.name,
                available=True,
                message="Ollama is available.",
                version=str(data.get("version", "")) or None,
            )
        except CortexMuxError as exc:
            return HealthStatus(provider=self.name, available=False, message=exc.message)

    async def list_models(self) -> list[ModelInfo]:
        """List installed Ollama models."""
        data = await self.client.get(OllamaClient.TAGS)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise ProviderResponseError("Ollama model list is invalid.", provider=self.name)
        result: list[ModelInfo] = []
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            modified = item.get("modified_at")
            result.append(
                ModelInfo(
                    name=item["name"],
                    provider=self.name,
                    size_bytes=item.get("size") if isinstance(item.get("size"), int) else None,
                    modified_at=_parse_datetime(modified),
                    family=_family(item),
                )
            )
        return result

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe tasks implemented by the Ollama adapter."""
        return [
            ProviderCapability(
                provider=self.name,
                model=model,
                task_types=_TASKS,
                streaming=True,
                structured_output=True,
                image_input=True,
                batch=True,
            )
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return adapter-level support; model abilities are configured by the caller."""
        return task in _TASKS

    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Dispatch a normalized request to its native Ollama operation."""
        if not request.model:
            raise InvalidRequestError(
                "Ollama requests require a model.",
                provider=self.name,
                request_id=request.request_id,
            )
        if isinstance(request, TextGenerationRequest):
            return await self._generate(request)
        if isinstance(request, ChatRequest):
            return await self._chat(request)
        if isinstance(request, StructuredOutputRequest):
            return await self._structured(request)
        if isinstance(request, VisionRequest):
            return await self._vision(request)
        if isinstance(request, EmbeddingRequest):
            return await self._embed(request)
        raise UnsupportedTaskError(
            "Ollama does not implement this request type.",
            provider=self.name,
            task=request.task.value,
        )

    async def stream(
        self, request: TextGenerationRequest | ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental normalized stream chunks."""
        if not request.model:
            raise InvalidRequestError("Ollama streaming requires a model.")
        if isinstance(request, TextGenerationRequest):
            path = OllamaClient.GENERATE
            payload = self._generation_payload(request)
        else:
            path = OllamaClient.CHAT
            payload = self._chat_payload(request)
        payload["stream"] = True
        async with self.client.stream(path, payload, request_id=request.request_id) as items:
            async for item in items:
                message = item.get("message")
                content = (
                    str(message.get("content", ""))
                    if isinstance(message, dict)
                    else str(item.get("response", ""))
                )
                yield StreamChunk(
                    request_id=request.request_id,
                    provider=self.name,
                    model=request.model,
                    content=content,
                    done=bool(item.get("done", False)),
                    usage=usage_from_ollama(item),
                )

    async def _generate(self, request: TextGenerationRequest) -> TextResponse:
        data = await self.client.post(
            OllamaClient.GENERATE,
            self._generation_payload(request),
            request_id=request.request_id,
        )
        return TextResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            content=_required_string(data, "response", request.request_id),
            usage=usage_from_ollama(data),
        )

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        data = await self.client.post(
            OllamaClient.CHAT, self._chat_payload(request), request_id=request.request_id
        )
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError(
                "Ollama chat response has no message.",
                provider=self.name,
                request_id=request.request_id,
            )
        return ChatResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            content=_required_string(message, "content", request.request_id),
            role=str(message.get("role", "assistant")),
            usage=usage_from_ollama(data),
        )

    async def _structured(self, request: StructuredOutputRequest) -> StructuredResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "format": request.json_schema or "json",
        }
        if request.system:
            payload["system"] = request.system
        if request.think is not None:
            payload["think"] = request.think
        payload["options"] = request.options
        data = await self.client.post(OllamaClient.GENERATE, payload, request_id=request.request_id)
        content = _required_string(data, "response", request.request_id)
        try:
            parsed = json.loads(content)
            if request.json_schema:
                _validate_json_schema(parsed, request.json_schema)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StructuredOutputValidationError(
                "Ollama returned invalid structured output.",
                provider=self.name,
                request_id=request.request_id,
                safe_excerpt=content[:200],
            ) from exc
        return StructuredResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            content=content,
            parsed=parsed,
            usage=usage_from_ollama(data),
        )

    async def _vision(self, request: VisionRequest) -> VisionResponse:
        images = [_encode_image(value, request.max_image_size_mb) for value in request.images]
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt, "images": images}],
            "stream": False,
            "options": request.options,
        }
        data = await self.client.post(OllamaClient.CHAT, payload, request_id=request.request_id)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError(
                "Ollama vision response has no message.", request_id=request.request_id
            )
        return VisionResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            content=_required_string(message, "content", request.request_id),
            usage=usage_from_ollama(data),
        )

    async def _embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        data = await self.client.post(
            OllamaClient.EMBED,
            {"model": request.model, "input": request.inputs},
            request_id=request.request_id,
        )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not all(isinstance(row, list) for row in embeddings):
            raise ProviderResponseError(
                "Ollama embedding response is invalid.", request_id=request.request_id
            )
        return EmbeddingResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            embeddings=[[float(value) for value in row] for row in embeddings],
            usage=usage_from_ollama(data),
        )

    def _generation_payload(self, request: TextGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": request.options,
        }
        if request.system:
            payload["system"] = request.system
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        return payload

    def _chat_payload(self, request: ChatRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            item: dict[str, Any] = {"role": message.role.value, "content": message.content}
            if message.images:
                item["images"] = message.images
            messages.append(item)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": request.options,
        }
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        return payload

    async def close(self) -> None:
        """Close the Ollama HTTP client."""
        await self.client.close()


def _required_string(data: dict[str, Any], key: str, request_id: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ProviderResponseError(
            "Ollama response is missing expected text.", request_id=request_id, field=key
        )
    return value


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _family(item: dict[str, Any]) -> str | None:
    details = item.get("details")
    return str(details["family"]) if isinstance(details, dict) and details.get("family") else None


def _encode_image(value: Path | bytes | str, max_size_mb: int) -> str:
    limit = max_size_mb * 1024 * 1024
    if isinstance(value, bytes):
        data = value
    else:
        path = Path(value).expanduser()
        if path.exists():
            if not path.is_file():
                raise InvalidRequestError("Vision image path is not a file.", path=str(path))
            if path.stat().st_size > limit:
                raise InvalidRequestError("Vision image exceeds the configured size limit.")
            data = path.read_bytes()
        elif isinstance(value, Path):
            raise InvalidRequestError("Vision image file does not exist.", path=str(path))
        else:
            try:
                data = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidRequestError(
                    "Vision image string must be an existing path or valid base64."
                ) from exc
    if len(data) > limit:
        raise InvalidRequestError("Vision image exceeds the configured size limit.")
    return base64.b64encode(data).decode("ascii")


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    checks: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if expected in checks and not isinstance(value, checks[expected]):
        raise ValueError(f"{path} must be {expected}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required if isinstance(required, list) else []:
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child in properties.items():
                if key in value and isinstance(child, dict):
                    _validate_json_schema(value[key], child, f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_json_schema(item, schema["items"], f"{path}[{index}]")
