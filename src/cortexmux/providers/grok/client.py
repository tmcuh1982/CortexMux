"""Direct, bounded HTTP transport for the xAI Responses API."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from time import perf_counter
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


def validate_grok_base_url(base_url: str) -> str:
    """Reject credentials and query secrets, and require HTTPS outside loopback."""
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
            "Grok requires HTTPS (except loopback) without URL credentials, query or fragment.",
            provider="grok",
        )
    return base_url.rstrip("/")


@dataclass(frozen=True, slots=True)
class GrokResult:
    """Successful JSON response and non-sensitive transport diagnostics."""

    data: dict[str, Any]
    attempts: int
    duration_seconds: float


class GrokClient:
    """Async Grok client with header authentication and no redirect following."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout: float,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
        retry_max_delay_seconds: float = 2,
    ) -> None:
        self.base_url = validate_grok_base_url(base_url)
        if not api_key.strip():
            raise ConfigurationError("Grok API key is empty.", provider="grok")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        if timeout <= 0 or retry_base_delay_seconds < 0 or retry_max_delay_seconds < 0:
            raise ValueError("timeout must be positive and retry delays non-negative")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def list_models(self) -> dict[str, Any]:
        """Fetch accessible language models and their advertised aliases."""
        return (await self._request("GET", "/language-models")).data

    async def respond(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> GrokResult:
        """Execute one non-streaming Responses API request."""
        return await self._request(
            "POST", "/responses", payload=payload, request_id=request_id, timeout=timeout
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> GrokResult:
        started = perf_counter()
        for attempt in range(1, self._max_retries + 2):
            context = {"provider": "grok", "request_id": request_id, "attempts": attempt}
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
                if attempt <= self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise ProviderTimeoutError("Grok request timed out.", **context) from None
            except httpx.RequestError:
                if attempt <= self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise ProviderUnavailableError("Grok is unavailable.", **context) from None
            if (
                response.status_code in {408, 429} or 500 <= response.status_code <= 599
            ) and attempt <= self._max_retries:
                await self._backoff(attempt)
                continue
            if response.status_code == 404:
                raise ModelNotFoundError("Grok model or endpoint was not found.", **context)
            if not response.is_success:
                raise ProviderResponseError(
                    "Grok returned an error response.",
                    status_code=response.status_code,
                    **context,
                )
            try:
                data = response.json()
            except ValueError:
                raise ProviderResponseError("Grok returned malformed JSON.", **context) from None
            if not isinstance(data, dict):
                raise ProviderResponseError("Grok returned an unexpected JSON value.", **context)
            return GrokResult(
                data=data, attempts=attempt, duration_seconds=perf_counter() - started
            )
        raise AssertionError("Grok retry loop exhausted without a result")  # pragma: no cover

    async def _backoff(self, attempt: int) -> None:
        ceiling = min(
            self._retry_max_delay_seconds,
            self._retry_base_delay_seconds * 2 ** (attempt - 1),
        )
        if ceiling > 0:
            await asyncio.sleep(random.uniform(0, ceiling))

    async def close(self) -> None:
        """Close only the HTTP client owned by this instance."""
        if self._owns_client:
            await self._client.aclose()
