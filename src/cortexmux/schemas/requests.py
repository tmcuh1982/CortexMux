"""Typed provider-neutral request schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cortexmux.core.types import TaskType
from cortexmux.schemas.common import ChatMessage
from cortexmux.schemas.decisions import DecisionQuestion, validate_decision_state


class CortexRequest(BaseModel):
    """Fields shared by every routed request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    task: TaskType
    provider: str | None = None
    model: str | None = None
    model_profile: str | None = Field(default=None, min_length=1)
    timeout: float | None = Field(default=None, gt=0)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class TextGenerationRequest(CortexRequest):
    """Text completion request."""

    task: Literal[TaskType.TEXT_GENERATION] = TaskType.TEXT_GENERATION
    prompt: str = Field(min_length=1)
    system: str | None = None
    stream: bool = False
    keep_alive: str | int | None = None


class ChatRequest(CortexRequest):
    """Conversational request using normalized messages."""

    task: Literal[TaskType.CHAT] = TaskType.CHAT
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    keep_alive: str | int | None = None


class StructuredOutputRequest(CortexRequest):
    """JSON or JSON-Schema constrained generation request."""

    task: Literal[TaskType.STRUCTURED_OUTPUT] = TaskType.STRUCTURED_OUTPUT
    prompt: str = Field(min_length=1)
    json_schema: dict[str, Any] | None = None
    system: str | None = None
    think: bool | None = None


class DecisionRequest(CortexRequest):
    """Evaluate bounded state against typed, independent questions."""

    task: Literal[TaskType.DECISION] = TaskType.DECISION
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, DecisionQuestion] = Field(min_length=1, max_length=64)

    @field_validator("state")
    @classmethod
    def state_is_bounded_json(
        cls, value: str | dict[str, Any] | list[Any]
    ) -> str | dict[str, Any] | list[Any]:
        """Keep remote decision input JSON-compatible and bounded."""
        return validate_decision_state(value)

    @field_validator("questions")
    @classmethod
    def question_ids_are_bounded(
        cls, value: dict[str, DecisionQuestion]
    ) -> dict[str, DecisionQuestion]:
        """Prevent empty or excessive question identifiers."""
        if any(not key or len(key) > 100 for key in value):
            raise ValueError("decision question IDs must contain 1 to 100 characters")
        return value

    @model_validator(mode="after")
    def payload_is_bounded(self) -> DecisionRequest:
        """Bound the entire remote decision payload, including question criteria."""
        payload = {
            "state": self.state,
            "questions": {
                key: question.model_dump(mode="json", exclude_none=True)
                for key, question in self.questions.items()
            },
        }
        if len(json.dumps(payload, ensure_ascii=False)) > 120_000:
            raise ValueError("decision payload exceeds the character limit")
        return self


class VisionRequest(CortexRequest):
    """Image analysis request."""

    task: Literal[TaskType.VISION] = TaskType.VISION
    prompt: str = Field(min_length=1)
    images: list[Path | bytes | str] = Field(min_length=1, repr=False)
    max_image_size_mb: int = Field(default=20, gt=0)


class EmbeddingRequest(CortexRequest):
    """Single or batch embedding request."""

    task: Literal[TaskType.EMBEDDING] = TaskType.EMBEDDING
    inputs: list[str] = Field(min_length=1)


class ImageGenerationRequest(CortexRequest):
    """ComfyUI workflow execution request."""

    task: Literal[TaskType.IMAGE_GENERATION] = TaskType.IMAGE_GENERATION
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    workflow: str | Path | dict[str, Any]
    bindings: dict[str, dict[str, str]] = Field(default_factory=dict)
    expected_output_nodes: list[str] = Field(default_factory=list)
    checkpoint: str | None = None
    seed: int | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, gt=0)
    guidance: float | None = Field(default=None, gt=0)
    sampler: str | None = None
    scheduler: str | None = None
    output_dir: Path | None = None
    extra_inputs: dict[str, str | int | float | bool] = Field(default_factory=dict)
    strict_bindings: bool = True


class DataAnalysisRequest(CortexRequest):
    """Safe deterministic data-analysis request."""

    task: Literal[TaskType.DATA_ANALYSIS] = TaskType.DATA_ANALYSIS
    source: Any
    instruction: str | None = None
    engine: Literal["auto", "pandas", "polars", "duckdb"] = "auto"
    plan: Any | None = None
    interpretation_provider: str | None = None
    interpretation_model: str | None = None
    verify_calculations: bool = True
    require_verified_calculations: bool = False
    create_charts: bool = False
    strict_planning: bool = False

    @model_validator(mode="after")
    def verification_mode_is_consistent(self) -> DataAnalysisRequest:
        """Strict verification requires calculation verification to be enabled."""
        if self.require_verified_calculations and not self.verify_calculations:
            raise ValueError("require_verified_calculations requires verify_calculations")
        return self

    @field_validator("source")
    @classmethod
    def source_must_be_present(cls, value: Any) -> Any:
        """Reject an absent source while allowing supported dataframe objects."""
        if value is None:
            raise ValueError("source is required")
        return value


RequestUnion = Annotated[
    TextGenerationRequest
    | ChatRequest
    | StructuredOutputRequest
    | VisionRequest
    | EmbeddingRequest
    | ImageGenerationRequest
    | DataAnalysisRequest
    | DecisionRequest,
    Field(discriminator="task"),
]
