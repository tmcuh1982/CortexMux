"""Minimal direct client for stable OpenAI API paths."""

from __future__ import annotations

from typing import Any

import httpx

from cortexmux.core.exceptions import (
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


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
    ) -> None:
        self.base_url = base_url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def get(self, path: str) -> dict[str, Any]:
        """GET a JSON object."""
        return await self._request("GET", path)

    async def post(self, path: str, payload: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        """POST a JSON object."""
        return await self._request("POST", path, payload=payload, request_id=request_id)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "OpenAI request timed out.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "OpenAI is unavailable.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        if response.status_code == 404:
            raise ModelNotFoundError(
                "OpenAI model or endpoint was not found.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
            )
        if response.is_error:
            raise ProviderResponseError(
                "OpenAI returned an error response.",
                provider="openai",
                endpoint=endpoint,
                status_code=response.status_code,
                request_id=request_id,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "OpenAI returned malformed JSON.",
                provider="openai",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(
                "OpenAI returned an unexpected JSON value.",
                provider="openai",
                request_id=request_id,
            )
        return data

    async def close(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            await self._client.aclose()
