"""Direct, bounded HTTP transport for the Google AI Studio Gemini API."""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx

from cortexmux.core.exceptions import (
    ConfigurationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


def validate_gemini_base_url(base_url: str) -> str:
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
            "Gemini requires HTTPS (except loopback) without URL credentials, query or fragment.",
            provider="gemini",
        )
    return base_url.rstrip("/")


def validate_model_id(model: str) -> str:
    """Accept a bare model ID without path, query, or action injection."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise InvalidRequestError("Gemini requires a bare model ID.", provider="gemini")
    return model


@dataclass(frozen=True, slots=True)
class GeminiResult:
    """Successful JSON response and non-sensitive transport diagnostics."""

    data: dict[str, Any]
    attempts: int
    duration_seconds: float


class GeminiClient:
    """Async Gemini client with header authentication and no redirect following."""

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
        self.base_url = validate_gemini_base_url(base_url)
        if not api_key.strip():
            raise ConfigurationError("Gemini API key is empty.", provider="gemini")
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

    async def list_models(self, page_token: str | None = None) -> dict[str, Any]:
        """Fetch one model page without exposing pagination tokens in diagnostics."""
        params = {"pageSize": "1000"}
        if page_token is not None:
            params["pageToken"] = page_token
        return (await self._request("GET", "/models", params=params)).data

    async def generate(
        self,
        model: str,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> GeminiResult:
        """Execute a single non-streaming generateContent request."""
        model_id = validate_model_id(model)
        return await self._request(
            "POST",
            f"/models/{model_id}:generateContent",
            payload=payload,
            request_id=request_id,
            timeout=timeout,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> GeminiResult:
        started = perf_counter()
        for attempt in range(1, self._max_retries + 2):
            context = {"provider": "gemini", "request_id": request_id, "attempts": attempt}
            try:
                response = await self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                    params=params,
                    timeout=self._timeout if timeout is None else timeout,
                    follow_redirects=False,
                )
            except httpx.TimeoutException:
                if attempt <= self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise ProviderTimeoutError("Gemini request timed out.", **context) from None
            except httpx.RequestError:
                if attempt <= self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise ProviderUnavailableError("Gemini is unavailable.", **context) from None
            if (
                response.status_code in {408, 429} or 500 <= response.status_code <= 599
            ) and attempt <= self._max_retries:
                await self._backoff(attempt)
                continue
            if response.status_code == 404:
                raise ModelNotFoundError("Gemini model or endpoint was not found.", **context)
            if not response.is_success:
                raise ProviderResponseError(
                    "Gemini returned an error response.",
                    status_code=response.status_code,
                    **context,
                )
            try:
                data = response.json()
            except ValueError:
                raise ProviderResponseError("Gemini returned malformed JSON.", **context) from None
            if not isinstance(data, dict):
                raise ProviderResponseError("Gemini returned an unexpected JSON value.", **context)
            return GeminiResult(
                data=data, attempts=attempt, duration_seconds=perf_counter() - started
            )
        raise AssertionError("Gemini retry loop exhausted without a result")  # pragma: no cover

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
