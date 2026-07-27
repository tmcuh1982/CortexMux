"""Helpers for translating Ollama responses."""

from __future__ import annotations

from typing import Any

from cortexmux.schemas.common import UsageMetadata


def usage_from_ollama(data: dict[str, Any]) -> UsageMetadata | None:
    """Normalize token counts and timing when present."""
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    if prompt is None and completion is None:
        return None
    prompt_count = int(prompt or 0)
    completion_count = int(completion or 0)
    return UsageMetadata(
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        total_tokens=prompt_count + completion_count,
        load_duration_ns=_optional_int(data.get("load_duration")),
        evaluation_duration_ns=_optional_int(data.get("eval_duration")),
    )


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None
