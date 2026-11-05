"""Bounded HTTP transport for MLX chat completions."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from cortexmux.core.exceptions import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def validate_mlx_base_url(base_url: str) -> str:
    """Validate an explicit HTTP(S) MLX API URL without sensitive URL components."""
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        raise ConfigurationError("Invalid MLX URL or port.", provider="mlx") from None
    if port == 0:
        raise ConfigurationError("Invalid MLX port.", provider="mlx")
    if (
        not parsed.hostname
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ConfigurationError(
            "MLX requires an HTTP(S) /v1 URL without credentials, query or fragment.",
            provider="mlx",
        )
    return base_url.rstrip("/")


class MLXClient:
    """Instance-owned client for non-streaming MLX requests."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        provider_name: str = "mlx",
        timeout: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = validate_mlx_base_url(base_url)
        self._provider_name = provider_name
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._api_key = api_key
        self._timeout = timeout

    async def complete(
        self, payload: dict[str, Any], *, request_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """Return one completed JSON response without exposing error bodies."""
        context = {"provider": self._provider_name, "request_id": request_id}
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
                json=payload,
                timeout=self._timeout if timeout is None else timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise ProviderTimeoutError("MLX request timed out.", **context) from None
        except httpx.RequestError:
            raise ProviderUnavailableError("MLX is unavailable.", **context) from None
        if response.status_code == 404:
            raise ModelNotFoundError("MLX model or endpoint was not found.", **context)
        if not response.is_success:
            raise ProviderResponseError(
                "MLX returned an error response.",
                status_code=response.status_code,
                **context,
            )
        try:
            data = response.json()
        except ValueError:
            raise ProviderResponseError("MLX returned malformed JSON.", **context) from None
        if not isinstance(data, dict):
            raise ProviderResponseError("MLX returned an unexpected JSON value.", **context)
        return data

    async def close(self) -> None:
        """Close only a client created by this instance."""
        if self._owns_client:
            await self._client.aclose()
