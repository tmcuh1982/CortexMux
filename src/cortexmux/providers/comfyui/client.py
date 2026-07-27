"""Direct HTTP and optional WebSocket ComfyUI client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from cortexmux.core.exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    WorkflowExecutionError,
)


class ComfyUIClient:
    """Use ComfyUI's local API without its browser UI."""

    ROOT = "/"
    SYSTEM_STATS = "/system_stats"
    MODELS = "/models"
    PROMPT = "/prompt"
    HISTORY = "/history"
    VIEW = "/view"
    WS = "/ws"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers
        )

    async def get_json(self, path: str, *, request_id: str | None = None) -> Any:
        """GET and decode JSON."""
        return await self._request_json("GET", path, request_id=request_id)

    async def post_json(self, path: str, payload: dict[str, Any], *, request_id: str) -> Any:
        """POST and decode JSON."""
        return await self._request_json("POST", path, json_body=payload, request_id=request_id)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Any:
        endpoint = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "ComfyUI request timed out.", endpoint=endpoint, request_id=request_id
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "ComfyUI is unavailable.", endpoint=endpoint, request_id=request_id
            ) from exc
        if response.is_error:
            raise ProviderResponseError(
                "ComfyUI returned an error response.",
                endpoint=endpoint,
                status_code=response.status_code,
                request_id=request_id,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "ComfyUI returned malformed JSON.", endpoint=endpoint, request_id=request_id
            ) from exc

    async def submit(self, graph: dict[str, Any], *, client_id: str, request_id: str) -> str:
        """Submit an API-format workflow and return its prompt ID."""
        result = await self.post_json(
            self.PROMPT,
            {"prompt": graph, "client_id": client_id},
            request_id=request_id,
        )
        prompt_id = result.get("prompt_id") if isinstance(result, dict) else None
        if not isinstance(prompt_id, str):
            raise ProviderResponseError(
                "ComfyUI queue response has no prompt_id.", request_id=request_id
            )
        return prompt_id

    async def history(self, prompt_id: str, *, request_id: str) -> dict[str, Any] | None:
        """Return a completed history entry when available."""
        data = await self.get_json(f"{self.HISTORY}/{prompt_id}", request_id=request_id)
        if not isinstance(data, dict):
            raise ProviderResponseError(
                "ComfyUI history response is invalid.", request_id=request_id
            )
        entry = data.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    async def wait_for_history(
        self, prompt_id: str, *, request_id: str, timeout: float
    ) -> dict[str, Any]:
        """Poll history as a reliable completion fallback."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            entry = await self.history(prompt_id, request_id=request_id)
            if entry is not None:
                status = entry.get("status")
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise WorkflowExecutionError(
                        "ComfyUI workflow failed.", prompt_id=prompt_id, request_id=request_id
                    )
                if isinstance(entry.get("outputs"), dict):
                    return entry
            await asyncio.sleep(0.25)
        raise ProviderTimeoutError(
            "Timed out waiting for ComfyUI workflow.",
            prompt_id=prompt_id,
            request_id=request_id,
        )

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        client_id: str,
        request_id: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Monitor WebSocket progress, then use history as the completion authority."""
        try:
            async with asyncio.timeout(timeout):
                async for event in self.websocket_events(client_id):
                    event_type = event.get("type")
                    data = event.get("data")
                    if event_type in {"execution_error", "execution_interrupted"}:
                        raise WorkflowExecutionError(
                            "ComfyUI reported workflow execution failure.",
                            prompt_id=prompt_id,
                            request_id=request_id,
                        )
                    if (
                        event_type == "executing"
                        and isinstance(data, dict)
                        and data.get("prompt_id") == prompt_id
                        and data.get("node") is None
                    ):
                        break
        except (TimeoutError, OSError):
            pass
        return await self.wait_for_history(prompt_id, request_id=request_id, timeout=timeout)

    async def websocket_events(self, client_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield ComfyUI WebSocket events when the optional dependency is installed."""
        try:
            import websockets
        except ImportError:
            return
        async with websockets.connect(self._websocket_url(client_id)) as socket:
            async for message in socket:
                if isinstance(message, str):
                    try:
                        item = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        yield item

    async def download_image(
        self,
        *,
        filename: str,
        subfolder: str,
        image_type: str,
        request_id: str,
    ) -> bytes:
        """Download one generated image from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        endpoint = f"{self.base_url}{self.VIEW}"
        try:
            response = await self._client.get(self.VIEW, params=params)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "ComfyUI image download timed out.", endpoint=endpoint, request_id=request_id
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "ComfyUI image download failed.", endpoint=endpoint, request_id=request_id
            ) from exc
        if response.is_error:
            raise ProviderResponseError(
                "ComfyUI image download returned an error.",
                status_code=response.status_code,
                request_id=request_id,
            )
        return response.content

    def _websocket_url(self, client_id: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse(
            (scheme, parsed.netloc, self.WS, "", urlencode({"clientId": client_id}), "")
        )

    async def close(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            await self._client.aclose()
