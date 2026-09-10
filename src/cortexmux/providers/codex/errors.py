"""Stable sanitized Codex errors; raw server diagnostics are never attached."""

from typing import Any

from cortexmux.core.exceptions import CortexMuxError, ProviderResponseError


class CodexError(CortexMuxError):
    """Codex failure with a stable machine-readable reason in context."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Codex request failed: {reason}.", provider="codex", reason=reason)


def server_error(error: dict[str, Any]) -> CortexMuxError:
    """Classify structured server failures without retaining server text."""
    info = error.get(
        "codexErrorInfo",
        error.get("data", {}).get("codexErrorInfo")
        if isinstance(error.get("data"), dict)
        else None,
    )
    if info == "usageLimitExceeded":
        return CodexError("quota_reached")
    if info == "unauthorized":
        return CodexError("connection_expired")
    if info == "sandboxError":
        return CodexError("permission_required")
    if isinstance(info, dict):
        for value in info.values():
            if isinstance(value, dict) and value.get("httpStatusCode") == 401:
                return CodexError("connection_expired")
            if isinstance(value, dict) and value.get("httpStatusCode") == 429:
                return CodexError("quota_reached")
    if error.get("code") in {-32601, -32602}:
        return CodexError("incompatible_version")
    return ProviderResponseError("Codex returned a failed response.", provider="codex")
