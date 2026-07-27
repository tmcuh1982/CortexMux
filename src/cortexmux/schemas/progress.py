"""Normalized provider progress events and callback types."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from cortexmux.core.types import ProgressStage


class ProgressEvent(BaseModel):
    """A typed progress update emitted during provider execution."""

    request_id: str
    provider: str
    stage: ProgressStage
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str | None = None
    prompt_id: str | None = None
    client_id: str | None = None
    node_id: str | None = None
    current: int | float | None = None
    total: int | float | None = None
    progress: float | None = Field(default=None, ge=0, le=1)
    queue_remaining: int | None = Field(default=None, ge=0)
    raw_event_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


async def emit_progress(
    callback: ProgressCallback | None,
    event: ProgressEvent,
) -> None:
    """Invoke a synchronous or asynchronous progress callback."""
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result
