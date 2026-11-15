"""Minimal direct client for stable OpenAI API paths."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from cortexmux.core.exceptions import (
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


@dataclass(frozen=True, slots=True)
class OpenAIResponseResult:
    """One successful OpenAI response plus bounded-call diagnostics."""

    data: dict[str, Any]
    attempts: int
    duration_seconds: float
    provider_request_id: str | None


class OpenAIClient:
    """Asynchronous OpenAI client without a mandatory SDK dependency."""

    MODELS = "/models"
    RESPONSES = "/responses"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout: float,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
        retry_max_delay_seconds: float = 2.0,
    ) -> None:
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be between 0 and 10")
        if retry_base_delay_seconds < 0 or retry_max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        self.base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def get(self, path: str) -> dict[str, Any]:
        """GET a JSON object."""
        return (await self._request("GET", path)).data

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """POST a JSON object, preserving the 0.5.1 client return type."""
        return (
            await self.post_with_metadata(
                path,
                payload,
                request_id=request_id,
                timeout=timeout,
            )
        ).data

    async def post_with_metadata(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> OpenAIResponseResult:
        """POST a JSON object and return bounded-call diagnostics."""
        return await self._request(
            "POST",
            path,
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
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> OpenAIResponseResult:
        endpoint = f"{self.base_url}{path}"
        started = perf_counter()
        response: httpx.Response | None = None
        max_attempts = self._max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=payload,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except httpx.TimeoutException as exc:
                if attempt < max_attempts:
                    await self._backoff(attempt)
                    continue
                raise ProviderTimeoutError(
                    "OpenAI request timed out.",
                    provider="openai",
                    endpoint=endpoint,
                    request_id=request_id,
                    attempts=attempt,
                ) from exc
            except httpx.RequestError as exc:
                if attempt < max_attempts:
                    await self._backoff(attempt)
                    continue
                raise ProviderUnavailableError(
                    "OpenAI is unavailable.",
                    provider="openai",
                    endpoint=endpoint,
                    request_id=request_id,
                    attempts=attempt,
                ) from exc
            if not _is_retryable_status(response.status_code) or attempt == max_attempts:
                break
            await self._backoff(attempt)

        if response is None:  # pragma: no cover - defensive invariant
            raise ProviderUnavailableError("OpenAI is unavailable.", provider="openai")
        attempts = attempt
        if response.status_code == 404:
            raise ModelNotFoundError(
                "OpenAI model or endpoint was not found.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
                attempts=attempts,
            )
        if response.is_error:
            raise ProviderResponseError(
                "OpenAI returned an error response.",
                provider="openai",
                endpoint=endpoint,
                status_code=response.status_code,
                request_id=request_id,
                attempts=attempts,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "OpenAI returned malformed JSON.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
                attempts=attempts,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(
                "OpenAI returned an unexpected JSON value.",
                provider="openai",
                request_id=request_id,
                attempts=attempts,
            )
        return OpenAIResponseResult(
            data=data,
            attempts=attempts,
            duration_seconds=perf_counter() - started,
            provider_request_id=response.headers.get("x-request-id"),
        )

    async def _backoff(self, failed_attempt: int) -> None:
        """Wait for bounded exponential backoff with full jitter."""
        ceiling = min(
            self._retry_max_delay_seconds,
            self._retry_base_delay_seconds * (2 ** (failed_attempt - 1)),
        )
        if ceiling > 0:
            await asyncio.sleep(random.uniform(0, ceiling))

    async def close(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            await self._client.aclose()


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS_CODES or 500 <= status_code <= 599
