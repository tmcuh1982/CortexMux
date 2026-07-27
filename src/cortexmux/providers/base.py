"""Provider abstraction owned by CortexMux."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from cortexmux.core.capabilities import ProviderCapability
from cortexmux.core.types import TaskType
from cortexmux.schemas.common import HealthStatus, ModelInfo
from cortexmux.schemas.requests import CortexRequest
from cortexmux.schemas.responses import CortexResponse


class BaseProvider(ABC):
    """Interface implemented by all network and deterministic providers."""

    name: str

    @abstractmethod
    async def healthcheck(self) -> HealthStatus:
        """Return current provider availability without raising for downtime."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List normalized models or engines."""

    @abstractmethod
    async def get_capabilities(self, model: str | None = None) -> list[ProviderCapability]:
        """Describe configured or introspected capabilities."""

    @abstractmethod
    def supports(self, task: TaskType, model: str | None = None) -> bool:
        """Return whether this provider accepts the task and optional model."""

    @abstractmethod
    async def execute(self, request: CortexRequest) -> CortexResponse:
        """Execute one normalized request."""

    async def close(self) -> None:
        """Release provider resources."""
        return None

    async def __aenter__(self) -> BaseProvider:
        """Enter the provider context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close provider resources."""
        await self.close()
