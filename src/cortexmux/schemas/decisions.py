"""Typed questions and probability answers for decision models."""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NoulQuestion(BaseModel):
    """Ask whether one narrowly scoped statement is true."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: dict[str, str] | None = None

    @field_validator("criteria")
    @classmethod
    def criteria_are_boolean(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Keep optional explanations limited to the two possible answers."""
        if value is not None and (not value or not set(value) <= {"true", "false"}):
            raise ValueError("noul criteria may contain only true and false")
        return value


class ChoiceQuestion(BaseModel):
    """Choose one option from an application-defined set."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: dict[str, str | None] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    """Score content against an ordered rubric."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: list[str] = Field(min_length=2, max_length=10)


DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class NoulAnswer(BaseModel):
    """Probability that the answer to a binary question is yes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)


class _DistributionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probabilities: dict[str, float] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("probabilities")
    @classmethod
    def valid_distribution(cls, values: dict[str, float]) -> dict[str, float]:
        """Reject impossible or incomplete probability distributions."""
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("probabilities must be finite and between zero and one")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=0.01):
            raise ValueError("probabilities must sum to one")
        return values


class ChoiceAnswer(_DistributionAnswer):
    """Selected option and the probability of every option."""

    type: Literal["choice"]
    choice: str

    @model_validator(mode="after")
    def choice_is_present(self) -> ChoiceAnswer:
        """Require the selected option to appear in its distribution."""
        if self.choice not in self.probabilities:
            raise ValueError("choice is missing from probabilities")
        return self


class ScoreAnswer(_DistributionAnswer):
    """Weighted score and the probability of every rubric level."""

    type: Literal["score"]
    score: float = Field(allow_inf_nan=False)
    legend: dict[str, str] = Field(min_length=2)

    @model_validator(mode="after")
    def levels_match(self) -> ScoreAnswer:
        """Require each returned probability to describe a declared level."""
        if set(self.legend) != set(self.probabilities):
            raise ValueError("score levels do not match the legend")
        return self


DecisionAnswer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


def validate_decision_state(
    value: str | dict[str, Any] | list[Any],
) -> str | dict[str, Any] | list[Any]:
    """Accept only bounded JSON text, objects or arrays as decision state."""
    if not isinstance(value, (str, dict, list)):
        raise ValueError("decision state must be a string, object or array")
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("decision state must be JSON serializable") from exc
    if len(serialized) > 100_000:
        raise ValueError("decision state exceeds the character limit")
    return value
