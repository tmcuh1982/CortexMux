"""Bounded reads shared by existing HTTP clients, without transport ownership."""

from typing import Any

import httpx

from cortexmux.core.exceptions import ProviderResponseError


async def read_bounded(
    response: httpx.Response,
    *,
    context: dict[str, str],
    audit: dict[str, Any] | None = None,
    limit: int = 2_000_000,
) -> bytes:
    """Read at most limit decoded bytes and retain bounded partial failure evidence."""
    body = bytearray()
    complete = False
    if audit is not None:
        audit.update(
            status_code=response.status_code,
            server_version_header=response.headers.get("x-mlx-vlm-version"),
        )
    try:
        async for chunk in response.aiter_bytes(chunk_size=65536):
            remaining = limit - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                if audit is not None:
                    audit["response_body_truncated"] = True
                raise ProviderResponseError(
                    "Response exceeds size limit.", category="size", **context
                )
        complete = True
        return bytes(body)
    finally:
        if audit is not None:
            audit["response_body"] = bytes(body)
            audit["response_body_truncated"] = not complete
