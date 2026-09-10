"""Deterministic provider and model routing."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from time import perf_counter

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import ModelNotFoundError, ModelSelectionError, UnsupportedTaskError
from cortexmux.core.registry import ProviderRegistry
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import RoutingMetadata
from cortexmux.schemas.progress import ProgressCallback
from cortexmux.schemas.requests import CortexRequest
from cortexmux.schemas.responses import CortexResponse, StreamEvent


class Router:
    """Apply explicit, deterministic routing rules without provider special cases."""

    def __init__(self, registry: ProviderRegistry, config: CortexMuxConfig) -> None:
        self.registry = registry
        self.config = config

    def select(self, request: CortexRequest) -> tuple[BaseProvider, str | None, str]:
        """Select exactly one provider/model and return an auditable reason."""
        if request.provider:
            provider = self.registry.get(request.provider)
            model = request.model or self.config.model_for(request.task, provider.name)
            if request.model:
                reason = "explicit_provider_and_model"
            else:
                reason = "explicit_provider_with_configured_default"
                if model is None and request.task.value not in {
                    "image_generation",
                    "data_analysis",
                }:
                    raise ModelSelectionError(
                        "The requested provider has no default model for this task.",
                        provider=provider.name,
                        task=request.task.value,
                        request_id=request.request_id,
                    )
            self._ensure_support(provider, request, model)
            return provider, model, reason

        if request.model:
            candidates = self.registry.supporting(request.task, request.model)
            if len(candidates) != 1:
                raise ModelSelectionError(
                    "An explicit model must match exactly one provider.",
                    model=request.model,
                    providers=[candidate.name for candidate in candidates],
                    request_id=request.request_id,
                )
            return candidates[0], request.model, "explicit_model_unique_provider"

        profile = self.config.routing.profile_for(request.model_profile)
        if profile is not None:
            route = profile.route_for(request.task)
            if route is not None:
                provider = self.registry.get(route.provider)
                self._ensure_support(provider, request, route.model)
                return provider, route.model, "configured_model_profile"

        provider_name = self.config.routing.defaults.provider_for(request.task)
        if provider_name:
            provider = self.registry.get(provider_name)
            model = self.config.model_for(request.task, provider.name)
            if model is None and request.task.value not in {"image_generation", "data_analysis"}:
                raise ModelSelectionError(
                    "The default route has no configured model.",
                    provider=provider.name,
                    task=request.task.value,
                    request_id=request.request_id,
                )
            self._ensure_support(provider, request, model)
            return provider, model, "configured_task_default"
        raise ModelSelectionError(
            "No explicit or configured provider/model route exists.",
            task=request.task.value,
            request_id=request.request_id,
        )

    async def route(
        self,
        request: CortexRequest,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> CortexResponse:
        """Select and execute a request, adding complete routing metadata."""
        started_at = datetime.now(UTC)
        started_clock = perf_counter()
        provider, model, reason = self.select(request)
        await self._ensure_model_is_available(provider, model, request)
        selected_request = self._selected_request(request, provider, model, reason)
        if on_progress is None:
            response = await provider.execute(selected_request)
        else:
            response = await provider.execute_with_progress(selected_request, on_progress)
        finished_at = datetime.now(UTC)
        response.routing = RoutingMetadata(
            requested_provider=request.provider,
            selected_provider=provider.name,
            requested_model=request.model,
            selected_model=model,
            task=request.task,
            routing_reason=reason,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=perf_counter() - started_clock,
            request_id=request.request_id,
        )
        return response

    async def stream(self, request: CortexRequest) -> AsyncGenerator[StreamEvent, None]:
        """Select a provider/model and stream a normalized request."""
        provider, model, reason = self.select(request)
        await self._ensure_model_is_available(provider, model, request)
        selected_request = self._selected_request(request, provider, model, reason)
        stream = provider.stream(selected_request)
        try:
            async for event in stream:
                yield event
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    def _selected_request(
        self,
        request: CortexRequest,
        provider: BaseProvider,
        model: str | None,
        reason: str,
    ) -> CortexRequest:
        """Apply configured profile options without overriding explicit request options."""
        options = request.options
        if reason == "configured_model_profile":
            profile = self.config.routing.profile_for(request.model_profile)
            route = profile.route_for(request.task) if profile is not None else None
            if route is not None:
                options = {**route.options, **request.options}
        return request.model_copy(
            update={"provider": provider.name, "model": model, "options": options}
        )

    @staticmethod
    def _ensure_support(provider: BaseProvider, request: CortexRequest, model: str | None) -> None:
        if not provider.supports(request.task, model):
            raise UnsupportedTaskError(
                "Provider does not support this task/model.",
                provider=provider.name,
                task=request.task.value,
                model=model,
                request_id=request.request_id,
            )

    async def _ensure_model_is_available(
        self,
        provider: BaseProvider,
        model: str | None,
        request: CortexRequest,
    ) -> None:
        """Require the selected model to be present when project policy enables it."""
        if model is None or not self.config.routing.validate_model_availability:
            return
        installed = await provider.list_models()
        installed_names = {item.name for item in installed}
        if model not in installed_names:
            raise ModelNotFoundError(
                "The selected model is not installed for this provider.",
                provider=provider.name,
                model=model,
                task=request.task.value,
                request_id=request.request_id,
                installed_models=sorted(installed_names)[:20],
            )
