"""Bounded, opt-in HTTP client for deterministic web-page extraction."""

from __future__ import annotations

import asyncio
from urllib.parse import urljoin, urlparse

import httpx

from cortexmux.core.config import WebConfig
from cortexmux.core.exceptions import WebAccessDisabledError, WebFetchError
from cortexmux.core.security import validate_web_url
from cortexmux.web.parser import extract_html, normalize_plain_text
from cortexmux.web.schemas import WebPage

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTML_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_TYPES = {"text/plain"}


class WebPageFetcher:
    """Fetch and parse web pages without exposing network access to a model."""

    def __init__(
        self,
        config: WebConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout_seconds,
        )

    async def fetch(self, url: str) -> WebPage:
        """Retrieve one public page and return bounded visible content."""
        if not self.config.enabled:
            raise WebAccessDisabledError("Web access is disabled. Set web.enabled=true explicitly.")
        requested_url = url
        current_url = await self._validate(url)
        for redirect_count in range(self.config.max_redirects + 1):
            response = await self._request(current_url)
            try:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise WebFetchError(
                            "Web redirect did not include a Location header.",
                            status_code=response.status_code,
                        )
                    if redirect_count >= self.config.max_redirects:
                        raise WebFetchError(
                            "Web page exceeded the redirect limit.",
                            max_redirects=self.config.max_redirects,
                        )
                    current_url = await self._validate(urljoin(current_url, location))
                    continue
                if not 200 <= response.status_code < 300:
                    raise WebFetchError(
                        "Web server returned an unsuccessful status.",
                        status_code=response.status_code,
                        host=_url_host(current_url),
                    )
                media_type = _media_type(response.headers.get("content-type", ""))
                if media_type not in _HTML_TYPES | _TEXT_TYPES:
                    raise WebFetchError(
                        "Web response content type is not supported.",
                        content_type=media_type or "missing",
                    )
                payload = await self._bounded_body(response)
                encoding = response.charset_encoding or "utf-8"
                body = payload.decode(encoding, errors="replace")
                if media_type in _HTML_TYPES:
                    title, text, tables = extract_html(
                        body,
                        max_tables=self.config.max_tables,
                        max_table_rows=self.config.max_table_rows,
                        max_table_columns=self.config.max_table_columns,
                        max_table_cells=self.config.max_table_cells,
                    )
                else:
                    title = None
                    text = normalize_plain_text(body)
                    tables = []
                truncated = len(text) > self.config.max_text_characters
                if truncated:
                    text = text[: self.config.max_text_characters].rstrip()
                return WebPage(
                    requested_url=requested_url,
                    final_url=current_url,
                    status_code=response.status_code,
                    content_type=media_type,
                    title=title,
                    text=text,
                    tables=tables,
                    bytes_received=len(payload),
                    text_truncated=truncated,
                )
            finally:
                await response.aclose()
        raise WebFetchError("Web page could not be resolved after redirects.")

    async def _validate(self, url: str) -> str:
        return await asyncio.to_thread(
            validate_web_url,
            url,
            allowed_hosts=self.config.allowed_hosts,
            allowed_ports=self.config.allowed_ports,
            allow_private_hosts=self.config.allow_private_hosts,
        )

    async def _request(self, url: str) -> httpx.Response:
        try:
            request = self._client.build_request("GET", url)
            return await self._client.send(
                request,
                stream=True,
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise WebFetchError("Web request timed out.", host=_url_host(url)) from exc
        except httpx.RequestError as exc:
            raise WebFetchError("Web request failed.", host=_url_host(url)) from exc

    async def _bounded_body(self, response: httpx.Response) -> bytes:
        declared_size = response.headers.get("content-length")
        if declared_size is not None:
            try:
                if int(declared_size) > self.config.max_response_bytes:
                    raise WebFetchError(
                        "Web response exceeds the configured size limit.",
                        max_response_bytes=self.config.max_response_bytes,
                    )
            except ValueError as exc:
                raise WebFetchError("Web response has an invalid Content-Length.") from exc
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            received += len(chunk)
            if received > self.config.max_response_bytes:
                raise WebFetchError(
                    "Web response exceeds the configured size limit.",
                    max_response_bytes=self.config.max_response_bytes,
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            await self._client.aclose()


def _media_type(content_type: str) -> str:
    return content_type.partition(";")[0].strip().lower()


def _url_host(url: str) -> str | None:
    return urlparse(url).hostname
