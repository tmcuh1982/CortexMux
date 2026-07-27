"""Instance-scoped provider registry."""

from __future__ import annotations

import builtins

from cortexmux.core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from cortexmux.core.types import TaskType
from cortexmux.providers.base import BaseProvider
from cortexmux.schemas.common import HealthStatus


class ProviderRegistry:
    """Register, query, inspect, and close provider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider, *, replace: bool = False) -> None:
        """Register a provider, rejecting duplicates unless replacement is explicit."""
        if provider.name in self._providers and not replace:
            raise ProviderRegistrationError(
                "Provider is already registered.", provider=provider.name
            )
        self._providers[provider.name] = provider

    def unregister(self, name: str) -> BaseProvider:
        """Remove and return a provider."""
        try:
            return self._providers.pop(name)
        except KeyError as exc:
            raise ProviderNotFoundError("Provider is not registered.", provider=name) from exc

    def get(self, name: str) -> BaseProvider:
        """Retrieve a provider by stable name."""
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderNotFoundError("Provider is not registered.", provider=name) from exc

    def list(self) -> builtins.list[BaseProvider]:
        """List providers in deterministic name order."""
        return [self._providers[name] for name in sorted(self._providers)]

    def supporting(self, task: TaskType, model: str | None = None) -> builtins.list[BaseProvider]:
        """List providers declaring support for a task/model pair."""
        return [provider for provider in self.list() if provider.supports(task, model)]

    async def health(self) -> builtins.list[HealthStatus]:
        """Collect health reports from every provider."""
        return [await provider.healthcheck() for provider in self.list()]

    async def close(self) -> None:
        """Close every provider, attempting all cleanups."""
        first_error: BaseException | None = None
        for provider in self.list():
            try:
                await provider.close()
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
