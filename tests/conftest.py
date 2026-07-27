"""Shared deterministic test doubles."""

from __future__ import annotations

from typing import Any

import pytest

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import HealthStatus, ModelInfo
from cortexmux.schemas.requests import CortexRequest
from cortexmux.schemas.responses import CortexResponse, TextResponse


class StubProvider(BaseProvider):
    """Small provider double with observable cleanup."""

    def __init__(
        self,
        name: str,
        *,
        tasks: set[TaskType] | None = None,
        models: set[str] | None = None,
    ) -> None:
        self.name = name
        self.tasks = tasks or {TaskType.TEXT_GENERATION}
        self.models = models
        self.closed = False
        self.requests: list[CortexRequest] = []

    async def healthcheck(self) -> HealthStatus:
        return HealthStatus(provider=self.name, available=True, message="ok")

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=model, provider=self.name) for model in sorted(self.models or set())]

    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        return [
            ProviderCapability(provider=self.name, model=model, task_types=frozenset(self.tasks))
        ]

    def supports(self, task: TaskType, model: str | None = None) -> bool:
        return task in self.tasks and (self.models is None or model is None or model in self.models)

    async def execute(self, request: CortexRequest) -> CortexResponse:
        self.requests.append(request)
        return TextResponse(
            provider=self.name,
            model=request.model,
            request_id=request.request_id,
            content="stub",
        )

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def stub_provider() -> StubProvider:
    """Return a reusable provider double."""
    return StubProvider("stub")


@pytest.fixture
def api_graph() -> dict[str, Any]:
    """Return a minimal valid API-format ComfyUI graph."""
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "test"}},
    }
