"""Deterministic provider and model routing."""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

from cortexmux.core.config import CortexMuxConfig
from cortexmux.core.exceptions import ModelSelectionError, UnsupportedTaskError
from cortexmux.core.registry import ProviderRegistry
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import RoutingMetadata
from cortexmux.schemas.progress import ProgressCallback
from cortexmux.schemas.requests import CortexRequest
from cortexmux.schemas.responses import CortexResponse


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
        selected_request = request.model_copy(update={"provider": provider.name, "model": model})
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
