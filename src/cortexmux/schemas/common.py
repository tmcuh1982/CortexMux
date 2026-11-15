"""Common normalized schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from cortexmux.core.types import MessageRole, TaskType


class ChatMessage(BaseModel):
    """A provider-neutral chat message."""

    role: MessageRole
    content: str
    images: list[str] = Field(default_factory=list, repr=False)


class RoutingMetadata(BaseModel):
    """Auditable routing decision and request timing."""

    requested_provider: str | None = None
    selected_provider: str
    requested_model: str | None = None
    selected_model: str | None = None
    task: TaskType
    routing_reason: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    request_id: str


class UsageMetadata(BaseModel):
    """Normalized token or execution usage when supplied by a provider."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    evaluation_duration_ns: int | None = Field(default=None, ge=0)


class HealthStatus(BaseModel):
    """Provider health result."""

    provider: str
    available: bool
    message: str
    version: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelInfo(BaseModel):
    """Normalized model metadata."""

    name: str
    provider: str
    size_bytes: int | None = None
    modified_at: datetime | None = None
    family: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ImageArtifact(BaseModel):
    """Downloaded image metadata without embedded binary data."""

    path: str
    filename: str
    node_id: str
    sequence: int
    mime_type: str | None = None
    size_bytes: int | None = None


class SentToModelMetadata(BaseModel):
    """Bounded disclosure record for optional data interpretation."""

    schema_columns: list[str] = Field(default_factory=list)
    sample_rows: int = 0
    aggregate_items: int = 0
    redacted_columns: list[str] = Field(default_factory=list)
    character_count: int = 0


class FrozenModel(BaseModel):
    """Base for immutable request identifiers."""

    model_config = ConfigDict(frozen=True)
