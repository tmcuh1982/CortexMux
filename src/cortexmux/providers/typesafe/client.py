"""Bounded HTTP transport for the TypeSafe System One API."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

import httpx

from cortexmux.core.exceptions import (
    ConfigurationError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def validate_typesafe_base_url(base_url: str) -> str:
    """Require a clean API root and HTTPS outside loopback."""
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parsed.hostname
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or (parsed.scheme != "https" and not loopback)
        or (not loopback and port not in {None, 443})
        or port == -1
    ):
        raise ConfigurationError(
            "TypeSafe requires an HTTPS API root without URL credentials or query values.",
            provider="typesafe",
        )
    return base_url.rstrip("/")


class TypeSafeClient:
    """Instance-owned authenticated client with bounded retry behavior."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = validate_typesafe_base_url(base_url)
        if not api_key.strip():
            raise ConfigurationError("TypeSafe API key is empty.", provider="typesafe")
        self._api_key = api_key
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def list_models(self) -> dict[str, Any]:
        """Return the account's available TypeSafe models and aliases."""
        return await self._request("GET", "/v1/models")

    async def evaluate(
        self, payload: dict[str, Any], *, request_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """Evaluate typed questions against one bounded state."""
        return await self._request(
            "POST", "/v1/systemone", payload=payload, request_id=request_id, timeout=timeout
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        context = {"provider": "typesafe", "request_id": request_id}
        for attempt in range(2):
            try:
                response = await self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout if timeout is None else timeout,
                    follow_redirects=False,
                )
            except httpx.TimeoutException:
                raise ProviderTimeoutError("TypeSafe request timed out.", **context) from None
            except httpx.RequestError:
                raise ProviderUnavailableError("TypeSafe is unavailable.", **context) from None
            if response.status_code in {429, 529} and attempt == 0:
                await asyncio.sleep(0.25)
                continue
            if not response.is_success:
                raise ProviderResponseError(
                    "TypeSafe returned an error response.",
                    status_code=response.status_code,
                    **context,
                )
            try:
                data = response.json()
            except ValueError:
                raise ProviderResponseError(
                    "TypeSafe returned malformed JSON.", **context
                ) from None
            if not isinstance(data, dict):
                raise ProviderResponseError("TypeSafe returned an invalid JSON object.", **context)
            return data
        raise AssertionError("TypeSafe retry loop exhausted")  # pragma: no cover

    async def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()
