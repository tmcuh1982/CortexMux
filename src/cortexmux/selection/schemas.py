"""Typed contracts for deterministic model qualification."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cortexmux.core.types import TaskType

BenchmarkTask = Literal[TaskType.CHAT, TaskType.STRUCTURED_OUTPUT]
ProfileName = Literal["fast", "balanced", "quality"]


class ValidationKind(StrEnum):
    """Whitelisted validators available to benchmark cases."""

    EXACT_TEXT = "exact_text"
    EXACT_JSON = "exact_json"


class BenchmarkCase(BaseModel):
    """One bounded prompt with an objective expected result."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task: BenchmarkTask
    prompt: str = Field(min_length=1, max_length=20_000)
    validator: ValidationKind
    expected_text: str | None = None
    expected_json: dict[str, Any] | None = None
    json_schema: dict[str, Any] | None = None
    weight: float = Field(default=1, gt=0, le=100)

    @model_validator(mode="after")
    def expected_value_matches_validator(self) -> BenchmarkCase:
        """Require exactly the expected value needed by the chosen validator."""
        if self.validator is ValidationKind.EXACT_TEXT and self.expected_text is None:
            raise ValueError("exact_text requires expected_text")
        if self.validator is ValidationKind.EXACT_JSON:
            if self.expected_json is None or self.json_schema is None:
                raise ValueError("exact_json requires expected_json and json_schema")
            if self.task is not TaskType.STRUCTURED_OUTPUT:
                raise ValueError("exact_json requires the structured_output task")
        return self


def _default_tasks() -> set[BenchmarkTask]:
    return {TaskType.CHAT, TaskType.STRUCTURED_OUTPUT}


def _default_profiles() -> set[ProfileName]:
    return {"fast", "balanced", "quality"}


class CandidateConfiguration(BaseModel):
    """One provider/model/options combination to benchmark independently."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tasks: set[BenchmarkTask] = Field(default_factory=_default_tasks)
    profiles: set[ProfileName] = Field(default_factory=_default_profiles)
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    estimated_memory_bytes: int | None = Field(default=None, gt=0)
    input_cost_per_million: float = Field(default=0, ge=0)
    output_cost_per_million: float = Field(default=0, ge=0)


class ScoreWeights(BaseModel):
    """Normalized utility weights for one recommendation profile."""

    model_config = ConfigDict(extra="forbid")

    quality: float = Field(ge=0)
    latency: float = Field(ge=0)
    cost: float = Field(ge=0)
    memory: float = Field(ge=0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScoreWeights:
        """Keep scores comparable and auditable."""
        if abs(self.quality + self.latency + self.cost + self.memory - 1) > 1e-9:
            raise ValueError("score weights must sum to 1")
        return self


def default_profile_weights() -> dict[str, ScoreWeights]:
    """Return conservative defaults for latency, balance, and quality."""
    return {
        "fast": ScoreWeights(quality=0.35, latency=0.45, cost=0.15, memory=0.05),
        "balanced": ScoreWeights(quality=0.55, latency=0.25, cost=0.15, memory=0.05),
        "quality": ScoreWeights(quality=0.80, latency=0.10, cost=0.05, memory=0.05),
    }


class QualificationSuite(BaseModel):
    """Portable benchmark inputs for one project or server."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    candidates: list[CandidateConfiguration] = Field(min_length=1, max_length=100)
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=100)
    repetitions: int = Field(default=1, ge=1, le=5)
    profile_weights: dict[str, ScoreWeights] = Field(default_factory=default_profile_weights)
    minimum_quality: float = Field(default=1, ge=0, le=1)
    cost_reference_usd: float = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> QualificationSuite:
        """Reject ambiguous candidates and benchmark cases."""
        candidate_ids = [item.id for item in self.candidates]
        case_ids = [item.id for item in self.cases]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case ids must be unique")
        return self


class MachineProfile(BaseModel):
    """Non-sensitive machine characteristics relevant to local inference."""

    system: str
    release: str
    architecture: str
    processor: str | None = None
    cpu_count: int | None = Field(default=None, ge=1)
    total_memory_bytes: int | None = Field(default=None, gt=0)


class CaseQualification(BaseModel):
    """Observed outcome for one candidate/case repetition."""

    case_id: str
    task: TaskType
    repetition: int = Field(ge=1)
    passed: bool
    syntax_valid: bool | None = None
    schema_valid: bool | None = None
    expected_match: bool | None = None
    latency_seconds: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    observed_excerpt: str | None = Field(default=None, max_length=1_000)
    error: str | None = None


class CandidateQualification(BaseModel):
    """All deterministic evidence collected for one configuration."""

    candidate_id: str
    provider: str
    model: str
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    available: bool
    estimated_memory_bytes: int | None = Field(default=None, gt=0)
    memory_fit: float = Field(default=1, ge=0, le=1)
    cases: list[CaseQualification] = Field(default_factory=list)
    error: str | None = None


class ModelRecommendation(BaseModel):
    """Best validated candidate for one task and operating profile."""

    profile: ProfileName
    task: TaskType
    candidate_id: str
    provider: str
    model: str
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)
    score: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    latency_score: float = Field(ge=0, le=1)
    cost_score: float = Field(ge=0, le=1)
    memory_score: float = Field(ge=0, le=1)


class AdaptiveModelRoute(BaseModel):
    """Provider route that can be copied into a project routing profile."""

    provider: str
    model: str
    options: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class QualificationManifest(BaseModel):
    """Portable, auditable output used to configure a project or server."""

    version: Literal[1] = 1
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    machine: MachineProfile
    candidates: list[CandidateQualification]
    recommendations: list[ModelRecommendation]
    routing_profiles: dict[str, dict[TaskType, AdaptiveModelRoute]] = Field(default_factory=dict)
