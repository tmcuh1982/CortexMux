"""Bounded HTTP transport for DeepSeek Chat Completions."""

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


def validate_deepseek_base_url(base_url: str) -> str:
    """Require a credential-free HTTPS endpoint outside loopback."""
    parsed = urlsplit(base_url)
    if (
        not parsed.hostname
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ConfigurationError(
            "DeepSeek requires HTTPS outside loopback and a URL without credentials, "
            "query or fragment.",
            provider="deepseek",
        )
    return base_url.rstrip("/")


class DeepSeekClient:
    """Instance-owned client for non-streaming DeepSeek requests."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = validate_deepseek_base_url(base_url)
        if not api_key.strip():
            raise ConfigurationError("DeepSeek API key is empty.", provider="deepseek")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self._api_key = api_key
        self._timeout = timeout

    async def complete(
        self, payload: dict[str, Any], *, request_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        """Return one completed JSON response without exposing error bodies."""
        context = {"provider": "deepseek", "request_id": request_id}
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout if timeout is None else timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise ProviderTimeoutError("DeepSeek request timed out.", **context) from None
        except httpx.RequestError:
            raise ProviderUnavailableError("DeepSeek is unavailable.", **context) from None
        if response.status_code == 404:
            raise ModelNotFoundError("DeepSeek model or endpoint was not found.", **context)
        if not response.is_success:
            raise ProviderResponseError(
                "DeepSeek returned an error response.",
                status_code=response.status_code,
                **context,
            )
        try:
            data = response.json()
        except ValueError:
            raise ProviderResponseError("DeepSeek returned malformed JSON.", **context) from None
        if not isinstance(data, dict):
            raise ProviderResponseError("DeepSeek returned an unexpected JSON value.", **context)
        return data

    async def close(self) -> None:
        """Close only a client created by this instance."""
        if self._owns_client:
            await self._client.aclose()
