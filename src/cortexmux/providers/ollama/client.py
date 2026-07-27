"""Small direct client for stable Ollama API paths."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from cortexmux.core.exceptions import (
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class OllamaClient:
    """Direct asynchronous client with normalized transport failures."""

    TAGS = "/api/tags"
    VERSION = "/api/version"
    GENERATE = "/api/generate"
    CHAT = "/api/chat"
    EMBED = "/api/embed"
    SHOW = "/api/show"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers
        )

    async def get(self, path: str, *, request_id: str | None = None) -> dict[str, Any]:
        """GET a JSON object."""
        return await self._request("GET", path, request_id=request_id)

    async def post(self, path: str, payload: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        """POST a JSON object."""
        return await self._request("POST", path, json_body=payload, request_id=request_id)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Ollama request timed out.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "Ollama is unavailable.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        if response.status_code == 404:
            raise ModelNotFoundError(
                "Ollama model or endpoint was not found.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            )
        if response.is_error:
            raise ProviderResponseError(
                "Ollama returned an error response.",
                provider="ollama",
                endpoint=endpoint,
                status_code=response.status_code,
                request_id=request_id,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "Ollama returned malformed JSON.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(
                "Ollama returned an unexpected JSON value.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            )
        return data

    @asynccontextmanager
    async def stream(
        self, path: str, payload: dict[str, Any], *, request_id: str
    ) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
        """Open an incremental NDJSON response, closed safely on cancellation."""
        endpoint = f"{self.base_url}{path}"
        try:
            async with self._client.stream("POST", path, json=payload) as response:
                if response.is_error:
                    raise ProviderResponseError(
                        "Ollama returned an error stream.",
                        provider="ollama",
                        endpoint=endpoint,
                        status_code=response.status_code,
                        request_id=request_id,
                    )

                async def items() -> AsyncIterator[dict[str, Any]]:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ProviderResponseError(
                                "Ollama stream contained malformed NDJSON.",
                                provider="ollama",
                                endpoint=endpoint,
                                request_id=request_id,
                            ) from exc
                        if not isinstance(item, dict):
                            raise ProviderResponseError(
                                "Ollama stream item was not an object.",
                                provider="ollama",
                                endpoint=endpoint,
                                request_id=request_id,
                            )
                        yield item

                yield items()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Ollama stream timed out.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "Ollama stream is unavailable.",
                provider="ollama",
                endpoint=endpoint,
                request_id=request_id,
            ) from exc

    async def close(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            await self._client.aclose()
