"""Normalized response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from cortexmux.core.types import TaskType
from cortexmux.schemas.common import ImageArtifact, RoutingMetadata, UsageMetadata


class CortexResponse(BaseModel):
    """Fields shared by every provider response."""

    task: TaskType
    provider: str
    model: str | None = None
    request_id: str
    routing: RoutingMetadata | None = None
    usage: UsageMetadata | None = None
    raw_metadata: dict[str, Any] | None = None


class TextResponse(CortexResponse):
    """Normalized generated text."""

    task: Literal[TaskType.TEXT_GENERATION] = TaskType.TEXT_GENERATION
    content: str


class ChatResponse(CortexResponse):
    """Normalized assistant chat response."""

    task: Literal[TaskType.CHAT] = TaskType.CHAT
    content: str
    role: str = "assistant"


class StructuredResponse(CortexResponse):
    """Validated structured output."""

    task: Literal[TaskType.STRUCTURED_OUTPUT] = TaskType.STRUCTURED_OUTPUT
    content: str
    parsed: Any


class VisionResponse(CortexResponse):
    """Normalized image analysis output."""

    task: Literal[TaskType.VISION] = TaskType.VISION
    content: str


class EmbeddingResponse(CortexResponse):
    """Normalized embedding vectors."""

    task: Literal[TaskType.EMBEDDING] = TaskType.EMBEDDING
    embeddings: list[list[float]]


class ImageGenerationResponse(CortexResponse):
    """Downloaded ComfyUI image artifacts."""

    task: Literal[TaskType.IMAGE_GENERATION] = TaskType.IMAGE_GENERATION
    images: list[ImageArtifact]
    prompt_id: str


class DataAnalysisResponse(CortexResponse):
    """Deterministic analysis, optional interpretation, and artifacts."""

    task: Literal[TaskType.DATA_ANALYSIS] = TaskType.DATA_ANALYSIS
    engine: str
    profile: dict[str, Any]
    results: list[dict[str, Any]] = Field(default_factory=list)
    report_markdown: str
    warnings: list[str] = Field(default_factory=list)
    plan: dict[str, Any] | None = None
    chart_paths: list[str] = Field(default_factory=list)
    interpretation: str | None = None
    disclosure: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """An incremental provider stream item."""

    request_id: str
    provider: str
    model: str | None = None
    content: str = ""
    done: bool = False
    usage: UsageMetadata | None = None
    raw_metadata: dict[str, Any] | None = None
