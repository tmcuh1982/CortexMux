"""High-level synchronous and asynchronous CortexMux API."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import InvalidRequestError
from cortexmux.core.registry import ProviderRegistry
from cortexmux.core.router import Router
from cortexmux.core.security import validate_provider_url
from cortexmux.core.types import MessageRole, TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.providers.comfyui import ComfyUIClient, ComfyUIProvider
from cortexmux.providers.data import DataAnalysisProvider
from cortexmux.providers.ollama import OllamaClient, OllamaProvider
from cortexmux.schemas.common import ChatMessage, HealthStatus, ModelInfo
from cortexmux.schemas.requests import (
    ChatRequest,
    CortexRequest,
    DataAnalysisRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    StructuredOutputRequest,
    TextGenerationRequest,
    VisionRequest,
)
from cortexmux.schemas.responses import (
    ChatResponse,
    CortexResponse,
    DataAnalysisResponse,
    EmbeddingResponse,
    ImageGenerationResponse,
    StructuredResponse,
    TextResponse,
    VisionResponse,
)

ResponseT = TypeVar("ResponseT", bound=CortexResponse)
ModelT = TypeVar("ModelT", bound=BaseModel)
SyncT = TypeVar("SyncT")


class CortexMux:
    """One local-first interface for routed AI and deterministic data tasks."""

    def __init__(
        self,
        config: CortexMuxConfig | None = None,
        *,
        register_builtin_providers: bool = True,
    ) -> None:
        self.config = config or CortexMuxConfig()
        self.registry = ProviderRegistry()
        self.router = Router(self.registry, self.config)
        self._closed = False
        if register_builtin_providers:
            self._register_builtins()

    @classmethod
    def from_env(
        cls,
        *,
        config_path: Path | str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> CortexMux:
        """Construct from TOML, environment variables, and explicit overrides."""
        return cls(CortexMuxConfig.load(path=config_path, overrides=overrides))

    def _register_builtins(self) -> None:
        core = self.config.core
        if self.config.providers.ollama.enabled:
            ollama_settings = self.config.providers.ollama
            url = validate_provider_url(
                ollama_settings.base_url,
                allow_remote_hosts=core.allow_remote_hosts,
                approved_hosts=core.approved_hosts,
            )
            ollama_client = OllamaClient(
                url,
                timeout=ollama_settings.timeout_seconds,
                headers={
                    key: value.get_secret_value() for key, value in ollama_settings.headers.items()
                },
            )
            self.registry.register(OllamaProvider(ollama_client))
        if self.config.providers.comfyui.enabled:
            comfyui_settings = self.config.providers.comfyui
            url = validate_provider_url(
                comfyui_settings.base_url,
                allow_remote_hosts=core.allow_remote_hosts,
                approved_hosts=core.approved_hosts,
            )
            comfyui_client = ComfyUIClient(
                url,
                timeout=comfyui_settings.timeout_seconds,
                headers={
                    key: value.get_secret_value() for key, value in comfyui_settings.headers.items()
                },
            )
            self.registry.register(
                ComfyUIProvider(
                    comfyui_client,
                    workflow_dir=comfyui_settings.workflow_dir,
                    output_dir=core.output_dir / "images",
                    model_folders=comfyui_settings.model_folders,
                    timeout=comfyui_settings.timeout_seconds,
                )
            )
        self.registry.register(
            DataAnalysisProvider(
                self.config.data,
                output_dir=core.output_dir,
                request_executor=self._execute_nested,
            )
        )

    async def _execute_nested(self, request: CortexRequest) -> CortexResponse:
        if request.provider is None:
            return await self.router.route(request)
        return await self.registry.get(request.provider).execute(request)

    def register_provider(self, provider: BaseProvider, *, replace: bool = False) -> None:
        """Register a custom provider on this facade instance."""
        self.registry.register(provider, replace=replace)

    async def arun(self, request: CortexRequest | TaskType | str, **fields: Any) -> CortexResponse:
        """Validate and execute a generic asynchronous request."""
        normalized = (
            request if isinstance(request, CortexRequest) else _request_for(request, fields)
        )
        return await self.router.route(normalized)

    def run(self, request: CortexRequest | TaskType | str, **fields: Any) -> CortexResponse:
        """Validate and execute a generic synchronous request."""
        return self._sync(self.arun(request, **fields))

    async def agenerate(
        self, prompt: str, *, provider: str | None = None, model: str | None = None, **options: Any
    ) -> TextResponse:
        """Generate text asynchronously."""
        response = await self.arun(
            TextGenerationRequest(prompt=prompt, provider=provider, model=model, options=options)
        )
        return cast(TextResponse, response)

    def generate(
        self, prompt: str, *, provider: str | None = None, model: str | None = None, **options: Any
    ) -> TextResponse:
        """Generate text synchronously."""
        return self._sync(self.agenerate(prompt, provider=provider, model=model, **options))

    async def achat(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChatMessage] | None = None,
        provider: str | None = None,
        model: str | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Run a chat request asynchronously."""
        normalized = messages or (
            [ChatMessage(role=MessageRole.USER, content=prompt)] if prompt else None
        )
        if not normalized:
            raise InvalidRequestError("chat requires prompt or messages")
        response = await self.arun(
            ChatRequest(messages=normalized, provider=provider, model=model, options=options)
        )
        return cast(ChatResponse, response)

    def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[ChatMessage] | None = None,
        provider: str | None = None,
        model: str | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Run a chat request synchronously."""
        return self._sync(
            self.achat(prompt, messages=messages, provider=provider, model=model, **options)
        )

    async def astructured(
        self,
        prompt: str,
        *,
        response_model: type[ModelT] | None = None,
        json_schema: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> StructuredResponse:
        """Generate and validate structured JSON asynchronously."""
        schema = response_model.model_json_schema() if response_model else json_schema
        response = cast(
            StructuredResponse,
            await self.arun(
                StructuredOutputRequest(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    json_schema=schema,
                )
            ),
        )
        if response_model:
            response.parsed = response_model.model_validate(response.parsed)
        return response

    def structured(
        self,
        prompt: str,
        *,
        response_model: type[ModelT] | None = None,
        json_schema: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> StructuredResponse:
        """Generate and validate structured JSON synchronously."""
        return self._sync(
            self.astructured(
                prompt,
                response_model=response_model,
                json_schema=json_schema,
                provider=provider,
                model=model,
            )
        )

    async def avision(
        self,
        image: Path | bytes | str | list[Path | bytes | str],
        *,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> VisionResponse:
        """Analyze one or more images asynchronously."""
        images = image if isinstance(image, list) else [image]
        return cast(
            VisionResponse,
            await self.arun(
                VisionRequest(prompt=prompt, images=images, provider=provider, model=model)
            ),
        )

    def vision(
        self,
        image: Path | bytes | str | list[Path | bytes | str],
        *,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> VisionResponse:
        """Analyze one or more images synchronously."""
        return self._sync(self.avision(image, prompt=prompt, provider=provider, model=model))

    async def aembed(
        self,
        inputs: str | list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Create one or more embeddings asynchronously."""
        values = [inputs] if isinstance(inputs, str) else inputs
        return cast(
            EmbeddingResponse,
            await self.arun(EmbeddingRequest(inputs=values, provider=provider, model=model)),
        )

    def embed(
        self,
        inputs: str | list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Create one or more embeddings synchronously."""
        return self._sync(self.aembed(inputs, provider=provider, model=model))

    async def agenerate_image(
        self,
        prompt: str,
        *,
        workflow: str | Path | dict[str, Any] | None = None,
        provider: str | None = "comfyui",
        model: str | None = None,
        **fields: Any,
    ) -> ImageGenerationResponse:
        """Run a ComfyUI workflow asynchronously."""
        selected_workflow = workflow or self.config.providers.comfyui.default_workflow
        if selected_workflow is None:
            raise InvalidRequestError("image generation requires a workflow")
        return cast(
            ImageGenerationResponse,
            await self.arun(
                ImageGenerationRequest(
                    prompt=prompt,
                    workflow=selected_workflow,
                    provider=provider,
                    model=model,
                    **fields,
                )
            ),
        )

    def generate_image(
        self,
        prompt: str,
        *,
        workflow: str | Path | dict[str, Any] | None = None,
        provider: str | None = "comfyui",
        model: str | None = None,
        **fields: Any,
    ) -> ImageGenerationResponse:
        """Run a ComfyUI workflow synchronously."""
        return self._sync(
            self.agenerate_image(
                prompt, workflow=workflow, provider=provider, model=model, **fields
            )
        )

    async def aanalyze_data(
        self,
        source: Any,
        *,
        instruction: str | None = None,
        engine: str = "auto",
        provider: str | None = "data",
        **fields: Any,
    ) -> DataAnalysisResponse:
        """Run deterministic data analysis asynchronously."""
        return cast(
            DataAnalysisResponse,
            await self.arun(
                DataAnalysisRequest(
                    source=source,
                    instruction=instruction,
                    engine=cast(Any, engine),
                    provider=provider,
                    **fields,
                )
            ),
        )

    def analyze_data(
        self,
        source: Any,
        *,
        instruction: str | None = None,
        engine: str = "auto",
        provider: str | None = "data",
        **fields: Any,
    ) -> DataAnalysisResponse:
        """Run deterministic data analysis synchronously."""
        return self._sync(
            self.aanalyze_data(
                source,
                instruction=instruction,
                engine=engine,
                provider=provider,
                **fields,
            )
        )

    async def alist_models(self, provider: str) -> list[ModelInfo]:
        """List a provider's models asynchronously."""
        return await self.registry.get(provider).list_models()

    def list_models(self, provider: str) -> list[ModelInfo]:
        """List a provider's models synchronously."""
        return self._sync(self.alist_models(provider))

    async def ahealth(self) -> list[HealthStatus]:
        """Return health for all providers asynchronously."""
        return await self.registry.health()

    def health(self) -> list[HealthStatus]:
        """Return health for all providers synchronously."""
        return self._sync(self.ahealth())

    async def aclose(self) -> None:
        """Close all provider resources once."""
        if not self._closed:
            await self.registry.close()
            self._closed = True

    def close(self) -> None:
        """Close all provider resources synchronously."""
        self._sync(self.aclose())

    @staticmethod
    def _sync(awaitable: Coroutine[Any, Any, SyncT]) -> SyncT:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        if hasattr(awaitable, "close"):
            awaitable.close()
        raise InvalidRequestError(
            "Synchronous CortexMux APIs cannot run inside an active event loop; "
            "use the corresponding async method."
        )

    async def __aenter__(self) -> CortexMux:
        """Enter an asynchronous facade context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources after an asynchronous context."""
        await self.aclose()

    def __enter__(self) -> CortexMux:
        """Enter a synchronous facade context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close resources after a synchronous context."""
        self.close()


def _request_for(task: TaskType | str, fields: dict[str, Any]) -> CortexRequest:
    try:
        task_type = TaskType(task)
    except ValueError as exc:
        raise InvalidRequestError("Unknown task type.", task=str(task)) from exc
    classes: dict[TaskType, type[CortexRequest]] = {
        TaskType.TEXT_GENERATION: TextGenerationRequest,
        TaskType.CHAT: ChatRequest,
        TaskType.STRUCTURED_OUTPUT: StructuredOutputRequest,
        TaskType.VISION: VisionRequest,
        TaskType.EMBEDDING: EmbeddingRequest,
        TaskType.IMAGE_GENERATION: ImageGenerationRequest,
        TaskType.DATA_ANALYSIS: DataAnalysisRequest,
    }
    try:
        return classes[task_type].model_validate({"task": task_type, **fields})
    except ValueError as exc:
        raise InvalidRequestError("Request validation failed.", task=task_type.value) from exc
