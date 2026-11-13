"""Bounded HTTP transport for MLX chat completions."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from cortexmux.core.bounded_http import read_bounded
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
        or parsed.path not in {"", "/", "/v1", "/v1/"}
    ):
        raise ConfigurationError(
            "MLX requires an HTTP(S) origin or /v1 URL without credentials, query or fragment.",
            provider="mlx",
        )
    return base_url.rstrip("/") + ("" if parsed.path.startswith("/v1") else "/v1")


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
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
        private_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one completed JSON response without exposing error bodies."""
        context = {"provider": self._provider_name, "request_id": request_id}
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
                json=payload,
                timeout=self._timeout if timeout is None else timeout,
                follow_redirects=False,
            ) as streamed:
                body = await read_bounded(streamed, context=context, audit=private_audit)
                # aiter_bytes already decodes Content-Encoding; do not decode it twice.
                response = httpx.Response(streamed.status_code, content=body)
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
        except (ValueError, RecursionError):
            raise ProviderResponseError("MLX returned malformed JSON.", **context) from None
        if not isinstance(data, dict):
            raise ProviderResponseError("MLX returned an unexpected JSON value.", **context)
        return data

    async def server_version(self, *, request_id: str, timeout: float | None) -> str | None:
        """Read the MLX-VLM version from the same origin's OpenAPI metadata."""
        context = {"provider": self._provider_name, "request_id": request_id}
        try:
            async with self._client.stream(
                "GET",
                f"{self.base_url.removesuffix('/v1')}/openapi.json",
                headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
                timeout=self._timeout if timeout is None else timeout,
                follow_redirects=False,
            ) as streamed:
                body = await read_bounded(streamed, context=context)
                response = httpx.Response(streamed.status_code, content=body)
        except httpx.TimeoutException:
            raise ProviderTimeoutError("MLX metadata request timed out.", **context) from None
        except httpx.RequestError:
            raise ProviderUnavailableError("MLX metadata is unavailable.", **context) from None
        if response.status_code in {401, 403, 404, 405}:
            return None
        if response.is_success:
            try:
                document = response.json()
                info = document.get("info") if isinstance(document, dict) else None
                if isinstance(info, dict) and info.get("title") == "MLX-VLM Inference API":
                    version = info.get("version")
                    if isinstance(version, str) and version:
                        return version
            except (ValueError, RecursionError):
                pass
        raise ProviderResponseError("Invalid MLX-VLM server metadata.", **context)

    async def close(self) -> None:
        """Close only a client created by this instance."""
        if self._owns_client:
            await self._client.aclose()
