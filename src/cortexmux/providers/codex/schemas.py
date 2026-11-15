"""Public, credential-free Codex account and request contracts."""

from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from pydantic import BaseModel, ConfigDict, Field


class CodexConfig(BaseModel):
    """Opt-in subscription transport with an application-owned auth directory."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    executable: str = Field(default="codex", min_length=1)
    auth_directory: Path = Field(default_factory=lambda: Path(user_data_dir("cortexmux")) / "codex")
    timeout_seconds: float = Field(default=120, gt=0)
    startup_timeout_seconds: float = Field(default=15, gt=0)
    shutdown_timeout_seconds: float = Field(default=5, gt=0)
    max_message_bytes: int = Field(default=2_000_000, ge=1024, le=20_000_000)
    max_output_characters: int = Field(default=1_000_000, gt=0)
    max_queued_events: int = Field(default=1024, gt=0)


class CodexRequestOptions(BaseModel):
    """Codex options; strict no-tool requests fail closed on the supported protocol."""

    model_config = ConfigDict(extra="forbid", strict=True)
    reasoning_effort: str | None = None
    require_no_tools: bool = True
    conversation_id: str | None = Field(default=None, min_length=1)
    persist_conversation: bool = False


class CodexAccount(BaseModel):
    """Allowlisted account fields, never authentication tokens."""

    connected: bool
    email: str | None = None
    plan_type: str | None = None


class CodexLogin(BaseModel):
    """Application-presented login ceremony; URLs and codes should not be logged."""

    login_id: str
    kind: Literal["chatgpt", "chatgptDeviceCode"]
    auth_url: str | None = Field(default=None, repr=False)
    verification_url: str | None = Field(default=None, repr=False)
    user_code: str | None = Field(default=None, repr=False)


class CodexLoginResult(BaseModel):
    """Sanitized terminal login result."""

    login_id: str
    success: bool


class CodexRateWindow(BaseModel):
    """Reported usage; absent values remain unknown, timestamps are Unix seconds."""

    model_config = ConfigDict(populate_by_name=True)
    used_percent: float | None = Field(default=None, alias="usedPercent")
    window_duration_mins: int | None = Field(default=None, alias="windowDurationMins")
    resets_at: int | None = Field(default=None, alias="resetsAt")


class CodexRateLimit(BaseModel):
    """One subscription usage bucket without inferred monetary costs."""

    model_config = ConfigDict(populate_by_name=True)
    limit_id: str | None = Field(default=None, alias="limitId")
    primary: CodexRateWindow | None = None
    secondary: CodexRateWindow | None = None


class CodexRateLimits(BaseModel):
    """Legacy and multi-bucket account limits, when available."""

    model_config = ConfigDict(populate_by_name=True)
    rate_limits: CodexRateLimit | None = Field(default=None, alias="rateLimits")
    by_limit_id: dict[str, CodexRateLimit] | None = Field(default=None, alias="rateLimitsByLimitId")
