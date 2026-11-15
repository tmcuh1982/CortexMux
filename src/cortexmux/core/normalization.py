"""Strict, provider-independent parsing of explicitly allowed JSON envelopes."""

import json
import math
import re
from typing import Any

_FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)


def parse_magistral(content: str) -> dict[str, Any]:
    """Parse one JSON object with an optional reasoning prefix and JSON fence."""
    return normalize_json_object(content, think_prefix=True, json_fence=True)[1]


def reject_constant(value: str) -> None:
    """Reject constants outside the JSON grammar."""
    raise ValueError("Non-finite constants are not valid JSON.")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON number exceeds the finite float range.")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def normalize_json_object(
    content: str, *, think_prefix: bool = False, json_fence: bool = False
) -> tuple[str, dict[str, Any], list[str]]:
    """Remove only explicitly allowed outer envelopes, preserving JSON text."""
    candidate, transformations = normalize_envelope(
        content, think_prefix=think_prefix, json_fence=json_fence
    )
    parsed = json.loads(
        candidate,
        object_pairs_hook=_unique_object,
        parse_constant=reject_constant,
        parse_float=_finite_float,
    )
    if not isinstance(parsed, dict):
        raise ValueError("Expected one JSON object.")
    return candidate, parsed, transformations


def normalize_envelope(
    content: str, *, think_prefix: bool = False, json_fence: bool = False
) -> tuple[str, list[str]]:
    """Return only syntax-authorized envelope removal, even before JSON validation."""
    candidate = content.strip()
    transformations: list[str] = []
    if think_prefix and candidate.startswith("[THINK]"):
        reasoning, closing, candidate = candidate[len("[THINK]") :].partition("[/THINK]")
        if not closing or "[THINK]" in reasoning:
            raise ValueError("Malformed reasoning prefix.")
        candidate = candidate.strip()
        transformations.append("think_prefix")
    fence = _FENCE.fullmatch(candidate) if json_fence else None
    if fence:
        candidate = fence[1]
        transformations.append("json_fence")
    return candidate, transformations
